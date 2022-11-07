"""PAR Synthesizer class."""

import inspect
import logging
import uuid

import numpy as np
import pandas as pd
import tqdm
from deepecho import PARModel
from deepecho.sequences import assemble_sequences

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table.base import BaseSynthesizer
from sdv.utils import cast_to_iterable

LOGGER = logging.getLogger(__name__)


class PARSynthesizer(BaseSynthesizer):
    """Synthesizer for sequential data.

    This synthesizer uses the ``deepecho.models.par.PARModel`` class as the core model.
    Additionally, it uses a separate synthesizer to model and sample the context columns
    to be passed into PAR.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        context_columns (list[str]):
            A list of strings, representing the columns that do not vary in a sequence.
        segment_size (int):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can be passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment.
        epochs (int):
            The number of epochs to train for. Defaults to 128.
        sample_size (int):
            The number of times to sample (before choosing and
            returning the sample which maximizes the likelihood).
            Defaults to 1.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'numerical': None,
        'boolean': None
    }

    def _get_context_metadata(self):
        context_columns_dict = {}
        context_columns = self.context_columns.copy() if self.context_columns else []
        if self._sequence_key:
            context_columns += self._sequence_key

        for column in context_columns:
            context_columns_dict[column] = self.metadata._columns[column]

        context_metadata_dict = {'columns': context_columns_dict}
        return SingleTableMetadata._load_from_dict(context_metadata_dict)

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=False,
                 context_columns=None, segment_size=None, epochs=128, sample_size=1, cuda=True,
                 verbose=False):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        sequence_key = self.metadata._sequence_key
        self._sequence_key = list(cast_to_iterable(sequence_key)) if sequence_key else None
        self._sequence_index = self.metadata._sequence_index
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.context_columns = context_columns
        self.segment_size = segment_size
        self._model_kwargs = {
            'epochs': epochs,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        context_metadata = self._get_context_metadata()
        self._context_synthesizer = GaussianCopulaSynthesizer(
            metadata=context_metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding
        )

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        for parameter_name, value in self._model_kwargs.items():
            instantiated_parameters[parameter_name] = value

        return instantiated_parameters

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        For PAR, none of the sequence keys are transformed.

        Args:
            data (pandas.DataFrame):
                The raw data to be transformed.

        Returns:
            pandas.DataFrame:
                The preprocessed data.
        """
        sequence_key_transformers = {sequence_key: None for sequence_key in self._sequence_key}
        if self._data_processor._hyper_transformer.field_transformers == {}:
            self.auto_assign_transformers(data)

        self.update_transformers(sequence_key_transformers)
        return super().preprocess(data)

    def _fit_context_model(self, transformed):
        LOGGER.debug(f'Fitting context synthesizer {self._context_synthesizer.__class__.__name__}')
        if self.context_columns:
            context = transformed[self._sequence_key + self.context_columns]
        else:
            context = transformed[self._sequence_key].copy()
            # Add constant column to allow modeling
            context[str(uuid.uuid4())] = 0

        context = context.groupby(self._sequence_key).first().reset_index()
        self._context_synthesizer.fit(context)

    def _transform_sequence_index(self, sequences):
        sequence_index_idx = self._data_columns.index(self._sequence_index)
        for sequence in sequences:
            data = sequence['data']
            sequence_index = data[sequence_index_idx]
            diffs = np.diff(sequence_index).tolist()
            data[sequence_index_idx] = diffs[0:1] + diffs
            data.append(sequence_index[0:1] * len(sequence_index))

    def _fit_sequence_columns(self, timeseries_data):
        self._model = PARModel(**self._model_kwargs)

        # handle output name from rdt
        if self._sequence_index:
            modified_name = self._sequence_index + '.value'
            if modified_name in timeseries_data.columns:
                timeseries_data = timeseries_data.rename(columns={
                    modified_name: self._sequence_index
                })

        self._output_columns = list(timeseries_data.columns)
        self._data_columns = [
            column
            for column in timeseries_data.columns
            if column not in self._sequence_key + self.context_columns
        ]

        sequences = assemble_sequences(
            timeseries_data,
            self._sequence_key,
            self.context_columns,
            self.segment_size,
            self._sequence_index,
            drop_sequence_index=False
        )
        data_types = []
        context_types = []
        for field in self._output_columns:
            dtype = timeseries_data[field].dtype
            kind = dtype.kind
            if kind in ('i', 'f'):
                data_type = 'continuous'
            elif kind in ('O', 'b'):
                data_type = 'categorical'
            else:
                raise ValueError(f'Unsupported dtype {dtype}')

            if field in self._data_columns:
                data_types.append(data_type)
            elif field in self.context_columns:
                context_types.append(data_type)

        if self._sequence_index:
            self._transform_sequence_index(sequences)
            data_types.append('continuous')

        # Validate and fit
        self._model.fit_sequences(sequences, context_types, data_types)

    def _fit(self, processed_data):
        """Fit this model to the data.

        Args:
            processed_data (pandas.DataFrame):
                pandas.DataFrame containing both the sequences,
                the entity columns and the context columns.
        """
        if self._sequence_key:
            self._fit_context_model(processed_data)

        LOGGER.debug(f'Fitting {self.__class__.__name__} model to table')
        self._fit_sequence_columns(processed_data)

    def _sample_from_par(self, context=None, sequence_length=None):
        """Sample new sequences.

        Args:
            context (pandas.DataFrame):
                Context values to use when generating the sequences.
                If not passed, the context values will be sampled
                using the specified tabular model.
            sequence_length (int):
                If passed, sample sequences of this length. If not
                given, the sequence length will be sampled from
                the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same
                format as that he training data had.
        """
        # Set the entity_columns as index to properly iterate over them
        if self._sequence_key:
            context = context.set_index(self._sequence_key)

        iterator = tqdm.tqdm(context.iterrows(), disable=not self._verbose, total=len(context))

        output = []
        for entity_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self._model.sample_sequence(context_values, sequence_length)
            if self._sequence_index:
                sequence_index_idx = self._data_columns.index(self._sequence_index)
                diffs = sequence[sequence_index_idx]
                start = sequence.pop(-1)
                sequence[sequence_index_idx] = np.cumsum(diffs) - diffs[0] + start

            # Reformat as a DataFrame
            group = pd.DataFrame(
                dict(zip(self._data_columns, sequence)),
                columns=self._data_columns
            )
            group[self._sequence_key] = entity_values
            for column, value in zip(self.context_columns, context_values):
                if column == self._sequence_index:
                    sequence_index = group[column]
                    group[column] = sequence_index.cumsum() - sequence_index.iloc[0] + value
                else:
                    group[column] = value

            output.append(group)

        output = pd.concat(output)
        output = output[self._output_columns].reset_index(drop=True)
        if self._sequence_index:
            output = output.rename(columns={
                self._sequence_index: self._sequence_index + '.value'
            })

        return output

    def _sample(self, context_columns, sequence_length=None, randomize_samples=False):
        self._randomize_samples(randomize_samples)
        sampled = self._sample_from_par(context_columns, sequence_length)
        return self._data_processor.reverse_transform(sampled)

    def sample(self, num_sequences, sequence_length=None, randomize_samples=False):
        """Sample new sequences.

        Args:
            num_sequences (int):
                Number of sequences to sample.
            sequence_length (int):
                If passed, sample sequences of this length. If ``None``, the sequence length will
                be sampled from the model.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults to False.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same format as the fitted data.
        """
        context_columns = self._context_synthesizer._sample_with_progress_bar(
            num_sequences, output_file_path='disable', show_progress_bar=False)

        for column in self._sequence_key or []:
            if column not in context_columns:
                context_columns[column] = range(len(context_columns))

        return self._sample(context_columns, sequence_length, randomize_samples)

    def sample_sequential_columns(self, context_columns, sequence_length=None,
                                  randomize_samples=False):
        """Sample the sequential columns based ont he provided context columns.

        Args:
            context_columns (pandas.DataFrame):
                Context values to use when generating the sequences.
            sequence_length (int):
                If passed, sample sequences of this length. If ``None``, the sequence length will
                be sampled from the model.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults to False.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences based on the provided context columns.
        """
        if not self._sequence_key:
            raise TypeError(
                'Cannot sample based on context columns if there is no sequence key. Please use '
                'PARSynthesizer.sample method instead.'
            )
        # reorder context columns
        context_columns = context_columns[self.context_columns]

        return self._sample(context_columns, sequence_length, randomize_samples)

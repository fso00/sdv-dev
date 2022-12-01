"""Base class for single table model presets."""

import logging
import sys

import cloudpickle
import rdt.transformers

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.utils import get_package_versions, throw_version_mismatch_warning

LOGGER = logging.getLogger(__name__)

FAST_ML_PRESET = 'FAST_ML'
PRESETS = {
    FAST_ML_PRESET: 'Use this preset to minimize the time needed to create a synthetic data model.'
}


class SingleTablePreset:
    """Class for all single table synthesizer presets.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            ``SingleTableMetadata`` instance.
        name (str):
            The preset to use.
    """

    _synthesizer = None
    _default_synthesizer = GaussianCopulaSynthesizer

    def _setup_fast_preset(self, metadata):
        self._synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            default_distribution='norm',
            enforce_rounding=False
        )
        self._synthesizer._data_processor._update_transformers_by_sdtypes(
            'categorical',
            rdt.transformers.FrequencyEncoder(add_noise=True)
        )

    def __init__(self, metadata, name):
        if name not in PRESETS:
            raise ValueError(f"'name' must be one of {PRESETS}.")

        self.name = name
        if name == FAST_ML_PRESET:
            self._setup_fast_preset(metadata)

    def fit(self, data):
        """Fit this model to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the model to.
        """
        self._synthesizer.fit(data)

    def sample(self, num_rows, randomize_samples=True, max_tries_per_batch=100,
               batch_size=None, output_file_path=None, conditions=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not
                write rows anywhere.
            conditions:
                Deprecated argument. Use the ``sample_conditions`` method with
                ``sdv.sampling.Condition`` objects instead.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = self._synthesizer.sample(
            num_rows, randomize_samples, max_tries_per_batch,
            batch_size, output_file_path, conditions
        )

        return sampled

    def sample_conditions(self, conditions, max_tries_per_batch=100, batch_size=None,
                          randomize_samples=True, output_file_path=None):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = self._synthesizer.sample_conditions(
            conditions, max_tries_per_batch, batch_size, randomize_samples, output_file_path)

        return sampled

    def sample_remaining_columns(self, known_columns, max_tries_per_batch=100, batch_size=None,
                                 randomize_samples=True, output_file_path=None):
        """Sample rows from this table.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries_per_batch (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = self._synthesizer.sample_remaining_columns(
            known_columns, max_tries_per_batch, batch_size, randomize_samples, output_file_path)

        return sampled

    def save(self, path):
        """Save this model instance to the given path using cloudpickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        self._package_versions = get_package_versions(getattr(self, '_synthesizer', None))

        with open(path, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a SingleTableSynthesizer instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            SingleTableSynthesizer:
                The loaded synthesizer.
        """
        with open(path, 'rb') as f:
            model = cloudpickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model

    @classmethod
    def list_available_presets(cls, out=sys.stdout):
        """List the available presets and their descriptions."""
        out.write(f'Available presets:\n{PRESETS}\n\n'
                  'Supply the desired preset using the `name` parameter.\n\n'
                  'Have any requests for custom presets? Contact the SDV team to learn '
                  'more an SDV Premium license.\n')

    def __repr__(self):
        """Represent single table preset instance as text.

        Returns:
            str
        """
        return f'SingleTablePreset(name={self.name})'

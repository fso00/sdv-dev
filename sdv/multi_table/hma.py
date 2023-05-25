"""Hierarchical Modeling Algorithms."""

import logging
import math
from copy import deepcopy

import numpy as np
import pandas as pd

from sdv.multi_table.base import BaseMultiTableSynthesizer

LOGGER = logging.getLogger(__name__)


class HMASynthesizer(BaseMultiTableSynthesizer):
    """Hierarchical Modeling Algorithm One.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
    """

    DEFAULT_SYNTHESIZER_KWARGS = {
        'default_distribution': 'beta'
    }

    def __init__(self, metadata, locales=None):
        super().__init__(metadata, locales=locales)
        self._table_sizes = {}
        self._max_child_rows = {}
        self._augmented_tables = []

    def _get_extension(self, child_name, child_table, foreign_key):
        """Generate the extension columns for this child table.

        The resulting dataframe will have an index that contains all the foreign key values.
        The values for a given index are generated by flattening a synthesizer fitted with
        the child rows with that foreign key value.

        Args:
            child_name (str):
                Name of the child table.
            child_table (pandas.DataFrame):
                Data for the child table.
            foreign_key (str):
                Name of the foreign key field.

        Returns:
            pandas.DataFrame
        """
        table_meta = self._table_synthesizers[child_name].get_metadata()

        extension_rows = []
        foreign_key_columns = self._get_all_foreign_keys(child_name)
        foreign_key_values = child_table[foreign_key].unique()
        child_table = child_table.set_index(foreign_key)

        index = []
        scale_columns = None
        for foreign_key_value in foreign_key_values:
            child_rows = child_table.loc[[foreign_key_value]]
            child_rows = child_rows[child_rows.columns.difference(foreign_key_columns)]

            try:
                if child_rows.empty:
                    row = pd.Series({'num_rows': len(child_rows)})
                    row.index = f'__{child_name}__{foreign_key}__' + row.index
                else:
                    synthesizer = self._synthesizer(
                        table_meta,
                        **self._table_parameters[child_name]
                    )
                    synthesizer.fit_processed_data(child_rows.reset_index(drop=True))
                    row = synthesizer._get_parameters()
                    row = pd.Series(row)
                    row.index = f'__{child_name}__{foreign_key}__' + row.index

                    if scale_columns is None:
                        scale_columns = [
                            column
                            for column in row.index
                            if column.endswith('scale')
                        ]

                    if len(child_rows) == 1:
                        row.loc[scale_columns] = None

                extension_rows.append(row)
                index.append(foreign_key_value)
            except Exception:
                # Skip children rows subsets that fail
                pass

        return pd.DataFrame(extension_rows, index=index)

    @staticmethod
    def _clear_nans(table_data):
        for column in table_data.columns:
            column_data = table_data[column]
            if column_data.dtype in (int, float):
                fill_value = 0 if column_data.isna().all() else column_data.mean()
            else:
                fill_value = column_data.mode()[0]

            table_data[column] = table_data[column].fillna(fill_value)

    def _get_foreign_keys(self, table_name, child_name):
        foreign_keys = []
        for relation in self.metadata.relationships:
            if table_name == relation['parent_table_name'] and\
               child_name == relation['child_table_name']:
                foreign_keys.append(deepcopy(relation['child_foreign_key']))

        return foreign_keys

    def _augment_table(self, table, tables, table_name):
        """Generate the extension columns for this table.

        For each of the table's foreign keys, generate the related extension columns,
        and extend the provided table.

        Args:
            table (pandas.DataFrame):
                The table to extend.
            tables (dict):
                A dictionary mapping table_name to table data (pandas.DataFrame).
            table_name (str):
                The name of the table.

        Returns:
            pandas.DataFrame:
                The extended table.
        """
        self._table_sizes[table_name] = len(table)
        LOGGER.info('Computing extensions for table %s', table_name)
        for child_name in self.metadata._get_child_map()[table_name]:
            if child_name not in self._augmented_tables:
                child_table = self._augment_table(tables[child_name], tables, child_name)

            else:
                child_table = tables[child_name]

            foreign_keys = self._get_foreign_keys(table_name, child_name)
            for index, foreign_key in enumerate(foreign_keys):
                extension = self._get_extension(child_name, child_table.copy(), foreign_key)
                table = table.merge(extension, how='left', right_index=True, left_index=True)
                num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
                table[num_rows_key] = table[num_rows_key].fillna(0)
                self._max_child_rows[num_rows_key] = table[num_rows_key].max()
                tables[table_name] = table

        self._augmented_tables.append(table_name)
        self._clear_nans(table)
        return table

    def _pop_foreign_keys(self, table_data, table_name):
        """Remove foreign keys from the ``table_data``.

        Args:
            table_data (pd.DataFrame):
                The table that contains the ``foreign_keys``.
            table_name (str):
                The name representing the table.

        Returns:
            keys (dict):
                A dictionary mapping with the foreign key and it's values within the table.
        """
        foreign_keys = self._get_all_foreign_keys(table_name)
        keys = {}
        for fk in foreign_keys:
            keys[fk] = table_data.pop(fk).to_numpy()

        return keys

    def _model_tables(self, augmented_data):
        """Model the augmented tables.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to an augmented ``pandas.DataFrame``.
        """
        parent_map = self.metadata._get_parent_map()
        for table_name, table in augmented_data.items():
            if table_name not in parent_map:
                keys = self._pop_foreign_keys(table, table_name)
                self._clear_nans(table)
                LOGGER.info('Fitting %s for table %s; shape: %s', self._synthesizer.__name__,
                            table_name, table.shape)

                if not table.empty:
                    self._table_synthesizers[table_name].fit_processed_data(table)

                for name, values in keys.items():
                    table[name] = values

    def _augment_tables(self, processed_data):
        """Fit this ``HMASynthesizer`` instance to the dataset data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        augmented_data = deepcopy(processed_data)
        self._augmented_tables = []
        parent_map = self.metadata._get_parent_map()
        for table_name in processed_data:
            if not parent_map.get(table_name):
                self._augment_table(augmented_data[table_name], augmented_data, table_name)

        LOGGER.info('Augmentation Complete')
        return augmented_data

    def _finalize(self, sampled_data):
        """Do the final touches to the generated data.

        This method reverts the previous transformations to go back
        to values in the original space and also adds the parent
        keys in case foreign key relationships exist between the tables.

        Args:
            sampled_data (dict):
                Generated data

        Return:
            pandas.DataFrame:
                Formatted synthesized data.
        """
        final_data = {}
        for table_name, table_rows in sampled_data.items():
            parents = self.metadata._get_parent_map().get(table_name)
            if parents:
                for parent_name in parents:
                    foreign_keys = self._get_foreign_keys(parent_name, table_name)
                    for foreign_key in foreign_keys:
                        if foreign_key not in table_rows:
                            parent_ids = self._find_parent_ids(
                                table_name,
                                parent_name,
                                foreign_key,
                                sampled_data
                            )
                            table_rows[foreign_key] = parent_ids.to_numpy()

            synthesizer = self._table_synthesizers.get(table_name)
            dtypes = synthesizer._data_processor._dtypes
            for name, dtype in dtypes.items():
                table_rows[name] = table_rows[name].dropna().astype(dtype)

            final_data[table_name] = table_rows[list(dtypes.keys())]

        return final_data

    def _extract_parameters(self, parent_row, table_name, foreign_key):
        """Get the params from a generated parent row.

        Args:
            parent_row (pandas.Series):
                A generated parent row.
            table_name (str):
                Name of the table to make the synthesizer for.
            foreign_key (str):
                Name of the foreign key used to form this
                parent child relationship.
        """
        prefix = f'__{table_name}__{foreign_key}__'
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key[len(prefix):] for key in keys}
        flat_parameters = parent_row[keys].fillna(0)

        num_rows_key = f'{prefix}num_rows'
        if num_rows_key in flat_parameters:
            num_rows = flat_parameters[num_rows_key]
            flat_parameters[num_rows_key] = min(
                self._max_child_rows[num_rows_key],
                math.ceil(num_rows)
            )

        return flat_parameters.rename(new_keys).to_dict()

    def _process_samples(self, table_name, sampled_rows):
        """Process the ``sampled_rows`` for the given ``table_name``.

        Process the raw samples and convert them to the original space by reverse transforming
        them. Also, when there are synthesizer columns (columns used to recreate an instance
        of a synthesizer), those will be returned together.
        """
        data_processor = self._table_synthesizers[table_name]._data_processor
        sampled = data_processor.reverse_transform(sampled_rows)

        synthesizer_columns = list(set(sampled_rows.columns) - set(sampled.columns))
        if synthesizer_columns:
            sampled = pd.concat([sampled, sampled_rows[synthesizer_columns]], axis=1)

        return sampled

    def _sample_rows(self, synthesizer, table_name, num_rows=None):
        """Sample ``num_rows`` from ``synthesizer``.

        Args:
            synthesizer (copula.multivariate.base):
                Fitted synthesizer.
            table_name (str):
                Name of the table to sample from.
            num_rows (int):
                Number of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        num_rows = num_rows or synthesizer._num_rows
        if synthesizer._model:
            sampled_rows = synthesizer._sample(num_rows)
        else:
            sampled_rows = pd.DataFrame(index=range(num_rows))

        return self._process_samples(table_name, sampled_rows)

    def _get_child_synthesizer(self, parent_row, table_name, foreign_key):
        parameters = self._extract_parameters(parent_row, table_name, foreign_key)
        table_meta = self.metadata.tables[table_name]
        synthesizer = self._synthesizer(table_meta, **self._table_parameters[table_name])
        synthesizer._set_parameters(parameters)

        return synthesizer

    def _sample_child_rows(self, table_name, parent_name, parent_row, sampled_data):
        """Sample child rows that reference the given parent row.

        The sampled rows will be stored in ``sampled_data`` under the ``table_name`` key.

        Args:
            table_name (str):
                The name of the table to sample.
            parent_name (str):
                The name of the parent table.
            parent_row (pandas.Series):
                The parent row the child rows should reference.
            sampled_data (dict):
                A map of table name to sampled table data (pandas.DataFrame).
        """
        foreign_key = self._get_foreign_keys(parent_name, table_name)[0]
        synthesizer = self._get_child_synthesizer(parent_row, table_name, foreign_key)
        table_rows = self._sample_rows(synthesizer, table_name)

        if len(table_rows):
            parent_key = self.metadata.tables[parent_name].primary_key
            table_rows[foreign_key] = parent_row[parent_key]

            previous = sampled_data.get(table_name)
            if previous is None:
                sampled_data[table_name] = table_rows
            else:
                sampled_data[table_name] = pd.concat(
                    [previous, table_rows]).reset_index(drop=True)

    def _sample_children(self, table_name, sampled_data, table_rows):
        """Recursively sample the child tables of the given table.

        Sampled child data will be stored into `sampled_data`.

        Args:
            table_name (str):
                The name of the table whose children will be sampled.
            sampled_data (dict):
                A map of table name to the sampled table data (pandas.DataFrame).
            table_rows (pandas.DataFrame):
                The sampled rows of the given table.
        """
        for child_name in self.metadata._get_child_map()[table_name]:
            if child_name not in sampled_data:
                LOGGER.info('Sampling rows from child table %s', child_name)
                for _, row in table_rows.iterrows():
                    self._sample_child_rows(child_name, table_name, row, sampled_data)

                child_rows = sampled_data[child_name]
                self._sample_children(child_name, sampled_data, child_rows)

    @staticmethod
    def _find_parent_id(likelihoods, num_rows):
        """Find the parent id for one row based on the likelihoods of parent id values.

        If likelihoods are invalid, fall back to the num_rows.

        Args:
            likelihoods (pandas.Series):
                The likelihood of parent id values.
            num_rows (pandas.Series):
                The number of times each parent id value appears in the data.

        Returns:
            int:
                The parent id for this row, chosen based on likelihoods.
        """
        mean = likelihoods.mean()
        if (likelihoods == 0).all():
            # All rows got 0 likelihood, fallback to num_rows
            likelihoods = num_rows
        elif pd.isna(mean) or mean == 0:
            # Some rows got singular matrix error and the rest were 0
            # Fallback to num_rows on the singular matrix rows and
            # keep 0s on the rest.
            likelihoods = likelihoods.fillna(num_rows)
        else:
            # at least one row got a valid likelihood, so fill the
            # rows that got a singular matrix error with the mean
            likelihoods = likelihoods.fillna(mean)

        total = likelihoods.sum()
        if total == 0:
            # Worse case scenario: we have no likelihoods
            # and all num_rows are 0, so we fallback to uniform
            length = len(likelihoods)
            weights = np.ones(length) / length
        else:
            weights = likelihoods.to_numpy() / total

        return np.random.choice(likelihoods.index.to_list(), p=weights)

    def _get_likelihoods(self, table_rows, parent_rows, table_name, foreign_key):
        """Calculate the likelihood of each parent id value appearing in the data.

        Args:
            table_rows (pandas.DataFrame):
                The rows in the child table.
            parent_rows (pandas.DataFrame):
                The rows in the parent table.
            table_name (str):
                The name of the child table.
            foreign_key (str):
                The foreign key column in the child table.

        Returns:
            pandas.DataFrame:
                A DataFrame of the likelihood of each parent id.
        """
        likelihoods = {}

        data_processor = self._table_synthesizers[table_name]._data_processor
        table_rows = data_processor.transform(table_rows)

        for parent_id, row in parent_rows.iterrows():
            parameters = self._extract_parameters(row, table_name, foreign_key)
            table_meta = self._table_synthesizers[table_name].get_metadata()
            synthesizer = self._synthesizer(table_meta, **self._table_parameters[table_name])
            synthesizer._set_parameters(parameters)
            try:
                with np.random.default_rng(np.random.get_state()[1]):
                    likelihoods[parent_id] = synthesizer._get_likelihood(table_rows)

            except (AttributeError, np.linalg.LinAlgError):
                likelihoods[parent_id] = None

        return pd.DataFrame(likelihoods, index=table_rows.index)

    def _find_parent_ids(self, table_name, parent_name, foreign_key, sampled_data):
        """Find parent ids for the given table and foreign key.

        The parent ids are chosen randomly based on the likelihood of the available
        parent ids in the parent table. If the parent table is not sampled, this method
        will first sample rows for the parent table.

        Args:
            table_name (str):
                The name of the table to find parent ids for.
            parent_name (str):
                The name of the parent table.
            foreign_key (str):
                The name of the foreign key column in the child table.
            sampled_data (dict):
                Map of table name to sampled data (pandas.DataFrame).

        Returns:
            pandas.Series:
                The parent ids for the given table data.
        """
        table_rows = sampled_data[table_name]
        if parent_name in sampled_data:
            parent_rows = sampled_data[parent_name]
        else:
            ratio = self._table_sizes[parent_name] / self._table_sizes[table_name]
            num_parent_rows = max(int(round(len(table_rows) * ratio)), 1)
            parent_model = self._table_synthesizers[parent_name]
            parent_rows = self._sample_rows(parent_model, parent_name, num_parent_rows)

        primary_key = self.metadata.tables[parent_name].primary_key
        parent_rows = parent_rows.set_index(primary_key)
        num_rows = parent_rows[f'__{table_name}__{foreign_key}__num_rows'].fillna(0).clip(0)

        likelihoods = self._get_likelihoods(table_rows, parent_rows, table_name, foreign_key)
        return likelihoods.apply(self._find_parent_id, axis=1, num_rows=num_rows)

    def _sample_table(self, table_name, scale=1.0, sample_children=True, sampled_data=None):
        """Sample a single table and optionally its children."""
        if sampled_data is None:
            sampled_data = {}

        num_rows = int(self._table_sizes[table_name] * scale)

        LOGGER.info('Sampling %s rows from table %s', num_rows, table_name)

        synthesizer = self._table_synthesizers[table_name]
        table_rows = self._sample_rows(synthesizer, table_name, num_rows)
        sampled_data[table_name] = table_rows

        if sample_children:
            self._sample_children(table_name, sampled_data, table_rows)

        return sampled_data

    def _sample(self, scale=1.0):
        """Sample the entire dataset.

        Returns a dictionary with all the tables of the dataset. The amount of rows sampled will
        depend from table to table. This is because the children tables are created modelling the
        relation that they have with their parent tables, so its behavior may change from one
        table to another.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled data tables as ``pandas.DataFrame``.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        sampled_data = {}
        for table in self.metadata.tables:
            if not self.metadata._get_parent_map().get(table):
                self._sample_table(table, scale=scale, sampled_data=sampled_data)

        return self._finalize(sampled_data)

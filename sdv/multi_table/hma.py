"""Hierarchical Modeling Algorithms."""

import logging
import math
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.sampling import BaseHierarchicalSampler

LOGGER = logging.getLogger(__name__)


class HMASynthesizer(BaseHierarchicalSampler, BaseMultiTableSynthesizer):
    """Hierarchical Modeling Algorithm One.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        verbose (bool):
            Whether to print progress for fitting or not.
    """

    DEFAULT_SYNTHESIZER_KWARGS = {
        'default_distribution': 'beta'
    }

    def __init__(self, metadata, locales=None, verbose=True):
        BaseMultiTableSynthesizer.__init__(self, metadata, locales=locales)
        self._table_sizes = {}
        self._max_child_rows = {}
        self._augmented_tables = []
        self._learned_relationships = 0
        self.verbose = verbose
        BaseHierarchicalSampler.__init__(
            self,
            self.metadata,
            self._table_synthesizers,
            self._table_sizes)

    def _num_extended_columns(self, table_name, parent_name, columns_per_table):
        """Get the number of columns that will be generated for table_name.

        A table generates:
            - 1 num_rows column for each for each foreign key with a specific parent
            - n*(n-1)/2 correlation columns for each data column
            - 4 parameters columns for each data column, with:
                - 1 column for parameter a
                - 1 column for parameter b
                - 1 column for parameter scale
                - 1 column for parameter loc
        """
        # num_rows columns
        # Since HMA only supports one relationship between two tables, this should always be 1
        num_cardinality_columns = len(self.metadata._get_foreign_keys(parent_name, table_name))

        # no parameter columns are generated if there are no data columns
        num_data_columns = columns_per_table[table_name]
        if num_data_columns == 0:
            return num_cardinality_columns

        num_correlation_columns = (num_data_columns - 1) * num_data_columns // 2
        num_parameters_columns = num_data_columns * 4

        return num_correlation_columns + num_cardinality_columns + num_parameters_columns

    def _estimate_columns_traversal(self, table_name, columns_per_table, visited):
        """Given a table, estimate how many columns each parent will model.

        This method recursiverly models the children of a table all the way to the leaf nodes.
        """
        for child_name in self.metadata._get_child_map()[table_name]:
            if child_name not in visited:
                self._estimate_columns_traversal(child_name, columns_per_table, visited)

            columns_per_table[table_name] += \
                self._num_extended_columns(child_name, table_name, columns_per_table)

        visited.add(table_name)

    def _estimate_number_of_modeled_columns(self):
        """Estimate the number of columns that will be modeled for each root table.

        This method estimates how many extended columns will be generated during the
        `_augment_tables` method, so it traverses the graph in the same way.
        If that method is ever changed, this should be updated to match.

        After running this method, `columns_per_table` will store an estimate of the
        total number of columns that each table has after running `_augment_tables`,
        that is, the number of extended columns generated by the child tables as well
        as the number of data columns in the table itself. Foreign keys and primary
        keys are not counted, since they are not modeled.

        Returns:
            dict:
                Dictionary of (table_name: int) mappings, indicating the estimated
                number of columns that will be modeled for each root table.
        """
        # This dict will store the number of data columns + extended columns for each table
        # Initialize it with the number of data columns per table
        columns_per_table = {}
        for table_name in self.metadata.tables:
            num_fks = len(self.metadata._get_all_foreign_keys(table_name))
            num_pks = 1
            total_cols = len(self.metadata.tables[table_name].columns)
            num_data_columns = total_cols - num_fks - num_pks
            columns_per_table[table_name] = num_data_columns

        # Starting at root tables, recursively estimate the number of columns
        # each table will model
        visited = set()
        non_root_tables = set(self.metadata._get_parent_map().keys())
        root_parents = set(self.metadata.tables.keys()) - non_root_tables
        for table_name in root_parents:
            self._estimate_columns_traversal(table_name, columns_per_table, visited)

        # Select only the root tables
        return {table_name: columns_per_table[table_name] for table_name in root_parents}

    def _get_extension(self, child_name, child_table, foreign_key, progress_bar_desc):
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
            progress_bar_desc (str):
                Progress bar description.

        Returns:
            pandas.DataFrame
        """
        table_meta = self._table_synthesizers[child_name].get_metadata()

        extension_rows = []
        foreign_key_columns = self.metadata._get_all_foreign_keys(child_name)
        foreign_key_values = child_table[foreign_key].unique()
        child_table = child_table.set_index(foreign_key)

        index = []
        scale_columns = None
        pbar_args = self._get_pbar_args(desc=progress_bar_desc)
        for foreign_key_value in tqdm(foreign_key_values, **pbar_args):
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

    def _augment_table(self, table, tables, table_name):
        """Recursively generate the extension columns for the tables in the graph.

        For each of the table's foreign keys, generate the related extension columns,
        and extend the provided table. Generate them first for the top level tables,
        then their children, and so on.

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
        child_map = self.metadata._get_child_map()[table_name]
        for child_name in child_map:
            if child_name not in self._augmented_tables:
                child_table = self._augment_table(tables[child_name], tables, child_name)
            else:
                child_table = tables[child_name]

            foreign_keys = self.metadata._get_foreign_keys(table_name, child_name)
            for foreign_key in foreign_keys:
                progress_bar_desc = (
                    f'({self._learned_relationships + 1}/{len(self.metadata.relationships)}) '
                    f"Tables '{table_name}' and '{child_name}' ('{foreign_key}')"
                )
                extension = self._get_extension(
                    child_name,
                    child_table.copy(),
                    foreign_key,
                    progress_bar_desc
                )
                table = table.merge(extension, how='left', right_index=True, left_index=True)
                num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
                table[num_rows_key] = table[num_rows_key].fillna(0)
                self._max_child_rows[num_rows_key] = table[num_rows_key].max()
                tables[table_name] = table
                self._learned_relationships += 1

        self._augmented_tables.append(table_name)
        self._clear_nans(table)
        return table

    def _augment_tables(self, processed_data):
        """Fit this ``HMASynthesizer`` instance to the dataset data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        augmented_data = deepcopy(processed_data)
        self._augmented_tables = []
        self._learned_relationships = 0
        parent_map = self.metadata._get_parent_map()
        self._print(text='Learning relationships:')
        for table_name in processed_data:
            if not parent_map.get(table_name):
                self._augment_table(augmented_data[table_name], augmented_data, table_name)

        LOGGER.info('Augmentation Complete')
        return augmented_data

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
        foreign_keys = self.metadata._get_all_foreign_keys(table_name)
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
        augmented_data_to_model = [
            (table_name, table)
            for table_name, table in augmented_data.items()
            if table_name not in parent_map
        ]
        self._print(text='\n', end='')
        pbar_args = self._get_pbar_args(desc='Modeling Tables')
        for table_name, table in tqdm(augmented_data_to_model, **pbar_args):
            keys = self._pop_foreign_keys(table, table_name)
            self._clear_nans(table)
            LOGGER.info('Fitting %s for table %s; shape: %s', self._synthesizer.__name__,
                        table_name, table.shape)

            if not table.empty:
                self._table_synthesizers[table_name].fit_processed_data(table)

            for name, values in keys.items():
                table[name] = values

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

    def _recreate_child_synthesizer(self, child_name, parent_name, parent_row):
        # When more than one foreign key exists between two tables, only the first one
        # will be modeled.
        foreign_key = self.metadata._get_foreign_keys(parent_name, child_name)[0]
        parameters = self._extract_parameters(parent_row, child_name, foreign_key)
        table_meta = self.metadata.tables[child_name]

        synthesizer = self._synthesizer(table_meta, **self._table_parameters[child_name])
        synthesizer._set_parameters(parameters)
        synthesizer._data_processor = self._table_synthesizers[child_name]._data_processor

        return synthesizer

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
                with np.random.Generator(np.random.get_state()[1]):
                    likelihoods[parent_id] = synthesizer._get_likelihood(table_rows)

            except (AttributeError, np.linalg.LinAlgError):
                likelihoods[parent_id] = None

        return pd.DataFrame(likelihoods, index=table_rows.index)

    def _find_parent_ids(self, child_table, parent_table, child_name, parent_name, foreign_key):
        """Find parent ids for the given table and foreign key.

        The parent ids are chosen randomly based on the likelihood of the available
        parent ids in the parent table.

        Args:
            child_table (pd.DataFrame):
                The child table dataframe.
            parent_table (pd.DataFrame):
                The parent table dataframe.
            child_name (str):
                The name of the child table.
            parent_name (dict):
                Map of table name to sampled data (pandas.DataFrame).
            foreign_key (str):
                The name of the foreign key column in the child table.

        Returns:
            pandas.Series:
                The parent ids for the given table data.
        """
        # Create a copy of the parent table with the primary key as index to calculate likelihoods
        primary_key = self.metadata.tables[parent_name].primary_key
        parent_table = parent_table.set_index(primary_key)
        num_rows = parent_table[f'__{child_name}__{foreign_key}__num_rows'].fillna(0).clip(0)

        likelihoods = self._get_likelihoods(child_table, parent_table, child_name, foreign_key)
        return likelihoods.apply(self._find_parent_id, axis=1, num_rows=num_rows)

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        for foreign_key in self.metadata._get_foreign_keys(parent_name, child_name):
            if foreign_key not in child_table:
                parent_ids = self._find_parent_ids(
                    child_table=child_table,
                    parent_table=parent_table,
                    child_name=child_name,
                    parent_name=parent_name,
                    foreign_key=foreign_key
                )
                child_table[foreign_key] = parent_ids.to_numpy()

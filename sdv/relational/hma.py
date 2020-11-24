"""Hierarchical Modeling Algorithms."""

import logging

import numpy as np
import pandas as pd

from sdv.relational.base import BaseRelationalModel
from sdv.tabular.copulas import GaussianCopula

LOGGER = logging.getLogger(__name__)


class HMA1(BaseRelationalModel):
    """Hierarchical Modeling Alrogirhtm One.

    Args:
        metadata (dict, str or Metadata):
            Metadata dict, path to the metadata JSON file or Metadata instance itself.
        root_path (str or None):
            Path to the dataset directory. If ``None`` and metadata is
            a path, the metadata location is used. If ``None`` and
            metadata is a dict, the current working directory is used.
        model (type):
            Class of the ``copula`` to use. Defaults to
            ``sdv.models.copulas.GaussianCopula``.
        model_kwargs (dict):
            Keyword arguments to pass to the model. If the default model is used, this
            defaults to using a ``gaussian`` distribution and a ``categorical_fuzzy``
            transformer.
    """

    DEFAULT_MODEL = GaussianCopula
    DEFAULT_MODEL_KWARGS = {
        'model': GaussianCopula,
        'model_kwargs': {
            'default_distribution': 'gaussian',
            'categorical_transformer': 'categorical_fuzzy',
        }
    }

    def __init__(self, metadata, root_path=None, model=None, model_kwargs=None):
        super().__init__(metadata, root_path)

        if model is None:
            model = self.DEFAULT_MODEL
            if model_kwargs is None:
                model_kwargs = self.DEFAULT_MODEL_KWARGS

        self._model = model
        self._model_kwargs = model_kwargs or {}
        self._models = {}
        self._table_sizes = {}

    # ######## #
    # MODELING #
    # ######## #

    def _get_extension(self, child_name, child_table, foreign_key):
        """Generate the extension columns for this child table.

        Each element of the list is generated for one single children.
        That dataframe should have as ``index.name`` the ``foreign_key`` name, and as index
        it's values.

        The values for a given index are generated by flattening a model fitted with
        the related data to that index in the children table.

        Args:
            child_name (str):
                Name of the child table.
            child_table (set[str]):
                Data for the child table.
            foreign_key (str):
                Name of the foreign key field.

        Returns:
            pandas.DataFrame
        """
        table_meta = self._models[child_name].get_metadata()

        extension_rows = list()
        foreign_key_values = child_table[foreign_key].unique()
        child_table = child_table.set_index(foreign_key)
        child_primary = self.metadata.get_primary_key(child_name)

        for foreign_key_value in foreign_key_values:
            child_rows = child_table.loc[[foreign_key_value]]
            if child_primary in child_rows.columns:
                del child_rows[child_primary]

            model = self._model(table_metadata=table_meta)
            model.fit(child_rows.reset_index(drop=True))
            row = model.get_parameters()
            row = pd.Series(row)
            row.index = '__' + child_name + '__' + row.index
            extension_rows.append(row)

        return pd.DataFrame(extension_rows, index=foreign_key_values)

    def _extend_table(self, table, tables, table_name):
        LOGGER.info('Computing extensions for table %s', table_name)
        for child_name in self.metadata.get_children(table_name):
            child_key = self.metadata.get_foreign_key(table_name, child_name)
            child_table = self._model_table(child_name, tables, child_key)
            extension = self._get_extension(child_name, child_table, child_key)
            table = table.merge(extension, how='left', right_index=True, left_index=True)
            table['__' + child_name + '__num_rows'].fillna(0, inplace=True)

        return table

    def _prepare_for_modeling(self, table_data, table_name, primary_key):
        table_meta = self.metadata.get_table_meta(table_name)
        table_meta['name'] = table_name

        fields = table_meta['fields']

        if primary_key:
            table_meta['primary_key'] = None
            del table_meta['fields'][primary_key]

        for column in table_data.columns:
            if column not in fields:
                fields[column] = {
                    'type': 'numerical',
                    'subtype': 'float'
                }

                column_data = table_data[column]
                if column_data.dtype in (np.int, np.float):
                    fill_value = column_data.mean()
                else:
                    fill_value = column_data.mode()[0]

                table_data[column] = table_data[column].fillna(fill_value)

        return table_meta

    def _model_table(self, table_name, tables, foreign_key=None):
        """Model the indicated table and its children.

        Args:
            table_name (str):
                Name of the table to model.
            tables (dict):
                Dict of original tables.
            foreign_key (str):
                Name of the foreign key that references this table. Used only when modeling
                a child table.

        Returns:
            pandas.DataFrame:
                table data with the extensions created while modeling its children.
        """
        LOGGER.info('Modeling %s', table_name)

        if tables:
            table = tables[table_name].copy()
        else:
            table = self.metadata.load_table(table_name)

        self._table_sizes[table_name] = len(table)

        primary_key = self.metadata.get_primary_key(table_name)
        if primary_key:
            table = table.set_index(primary_key)
            table = self._extend_table(table, tables, table_name)

        table_meta = self._prepare_for_modeling(table, table_name, primary_key)

        if foreign_key:
            foreign_key_values = table.pop(foreign_key).values
            del table_meta['fields'][foreign_key]

        LOGGER.info('Fitting %s for table %s; shape: %s', self._model.__name__,
                    table_name, table.shape)
        model = self._model(**self._model_kwargs, table_metadata=table_meta)
        model.fit(table)
        self._models[table_name] = model

        if primary_key:
            table.reset_index(inplace=True)

        if foreign_key:
            table[foreign_key] = foreign_key_values

        return table

    def _fit(self, tables=None):
        """Fit this HMA1 instance to the dataset data.

        Args:
            tables (dict):
                Dictionary with the table names as key and ``pandas.DataFrame`` instances as
                values.  If ``None`` is given, the tables will be loaded from the paths
                indicated in ``metadata``. Defaults to ``None``.
        """
        self.metadata.validate(tables)

        for table_name in self.metadata.get_tables():
            if not self.metadata.get_parents(table_name):
                self._model_table(table_name, tables)

        LOGGER.info('Modeling Complete')

    # ######## #
    # SAMPLING #
    # ######## #

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
        final_data = dict()
        for table_name, table_rows in sampled_data.items():
            parents = self.metadata.get_parents(table_name)
            if parents:
                for parent_name in parents:
                    foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
                    if foreign_key not in table_rows:
                        parent_ids = self._find_parent_ids(table_name, parent_name, sampled_data)
                        table_rows[foreign_key] = parent_ids

            fields = self.metadata.get_fields(table_name)

            final_data[table_name] = table_rows[list(fields.keys())]

        return final_data

    def _extract_parameters(self, parent_row, table_name):
        """Get the params from a generated parent row.

        Args:
            parent_row (pandas.Series):
                A generated parent row.
            table_name (str):
                Name of the table to make the model for.
        """
        prefix = '__{}__'.format(table_name)
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key[len(prefix):] for key in keys}
        flat_parameters = parent_row[keys]
        return flat_parameters.rename(new_keys).to_dict()

    def _sample_rows(self, model, table_name, num_rows=None):
        """Sample ``num_rows`` from ``model``.

        Args:
            model (copula.multivariate.base):
                Fitted model.
            table_name (str):
                Name of the table to sample from.
            num_rows (int):
                Number of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        sampled = model.sample(num_rows)
        primary_key_name = self.metadata.get_primary_key(table_name)
        if primary_key_name:
            primary_key_values = self._get_primary_keys(table_name, len(sampled))
            sampled[primary_key_name] = primary_key_values

        return sampled

    def _sample_children(self, table_name, sampled_data, table_rows=None):
        if table_rows is None:
            table_rows = sampled_data[table_name]

        for child_name in self.metadata.get_children(table_name):
            for _, row in table_rows.iterrows():
                self._sample_child_rows(child_name, table_name, row, sampled_data)

    def _sample_child_rows(self, table_name, parent_name, parent_row, sampled_data):
        parameters = self._extract_parameters(parent_row, table_name)

        table_meta = self._models[table_name].get_metadata()
        model = self._model(table_metadata=table_meta)
        model.set_parameters(parameters)

        table_rows = self._sample_rows(model, table_name)
        if not table_rows.empty:
            parent_key = self.metadata.get_primary_key(parent_name)
            foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
            table_rows[foreign_key] = parent_row[parent_key]

            previous = sampled_data.get(table_name)
            if previous is None:
                sampled_data[table_name] = table_rows
            else:
                sampled_data[table_name] = pd.concat(
                    [previous, table_rows]).reset_index(drop=True)

            self._sample_children(table_name, sampled_data, table_rows)

    @staticmethod
    def _find_parent_id(likelihoods, num_rows):
        mean = likelihoods.mean()
        if (likelihoods == 0).all():
            # All rows got 0 likelihood, fallback to num_rows
            likelihoods = num_rows
        elif pd.isnull(mean) or mean == 0:
            # Some rows got singlar matrix error and the rest were 0
            # Fallback to num_rows on the singular matrix rows and
            # keep 0s on the rest.
            likelihoods = likelihoods.fillna(num_rows)
        else:
            # at least one row got a valid likelihood, so fill the
            # rows that got a singular matrix error with the mean
            likelihoods = likelihoods.fillna(mean)

        weights = likelihoods.values / likelihoods.sum()

        return np.random.choice(likelihoods.index, p=weights)

    def _get_likelihoods(self, table_rows, parent_rows, table_name):
        likelihoods = dict()
        for parent_id, row in parent_rows.iterrows():
            parameters = self._extract_parameters(row, table_name)
            table_meta = self._models[table_name].get_metadata()
            model = self._model(table_metadata=table_meta)
            model.set_parameters(parameters)
            try:
                likelihoods[parent_id] = model.get_likelihood(table_rows)
            except np.linalg.LinAlgError:
                likelihoods[parent_id] = None

        return pd.DataFrame(likelihoods, index=table_rows.index)

    def _find_parent_ids(self, table_name, parent_name, sampled_data):
        table_rows = sampled_data[table_name]
        if parent_name in sampled_data:
            parent_rows = sampled_data[parent_name]
        else:
            ratio = self._table_sizes[parent_name] / self._table_sizes[table_name]
            num_parent_rows = max(int(round(len(table_rows) * ratio)), 1)
            parent_model = self._models[parent_name]
            parent_rows = self._sample_rows(parent_model, parent_name, num_parent_rows)

        primary_key = self.metadata.get_primary_key(parent_name)
        parent_rows = parent_rows.set_index(primary_key)
        num_rows = parent_rows['__' + table_name + '__num_rows'].fillna(0).clip(0)

        likelihoods = self._get_likelihoods(table_rows, parent_rows, table_name)
        return likelihoods.apply(self._find_parent_id, axis=1, num_rows=num_rows)

    def _sample_table(self, table_name, num_rows=None, sample_children=True):
        """Sample a single table and optionally its children."""
        if num_rows is None:
            num_rows = self._table_sizes[table_name]

        model = self._models[table_name]
        table_rows = self._sample_rows(model, table_name, num_rows)

        if sample_children:
            sampled_data = {
                table_name: table_rows
            }

            self._sample_children(table_name, sampled_data)
            return self._finalize(sampled_data)

        else:
            return self._finalize({table_name: table_rows})[table_name]

    def _sample(self, table_name=None, num_rows=None, sample_children=True):
        """Sample the entire dataset.

        ``sample_all`` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match ``num_rows`` on tables without parents.

        This is because the children tables are created modelling the relation that they have
        with their parent tables, so its behavior may change from one table to another.

        Args:
            num_rows (int):
                Number of rows to be sampled on the first parent tables. If ``None``,
                sample the same number of rows as in the original tables.
            reset_primary_keys (bool):
                Whether or not reset the primary key generators.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled datatables as ``pandas.DataFrame``.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        if table_name:
            return self._sample_table(table_name, num_rows, sample_children)

        sampled_data = dict()
        for table in self.metadata.get_tables():
            if not self.metadata.get_parents(table):
                sampled = self._sample_table(table, num_rows)
                sampled_data.update(sampled)

        return sampled_data

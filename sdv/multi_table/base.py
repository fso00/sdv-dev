"""Base Multi Table Synthesizer class."""

from collections import defaultdict
from copy import deepcopy

import pandas as pd

from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.errors import InvalidDataError


class BaseMultiTableSynthesizer:
    """Base class for multi table synthesizers.

    The ``BaseMultiTableSynthesizer`` class defines the common API that all the
    multi table synthesizers need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
    """

    _synthesizer = GaussianCopulaSynthesizer

    def _initialize_models(self):
        for table_name, table_metadata in self.metadata._tables.items():
            synthesizer_parameters = self._table_parameters.get(table_name, {})
            self._table_synthesizers[table_name] = self._synthesizer(
                metadata=table_metadata,
                **synthesizer_parameters
            )

    def __init__(self, metadata):
        self.metadata = metadata
        self.metadata.validate()
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)
        self._initialize_models()

    def get_table_parameters(self, table_name):
        """Return the parameters that will be used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters that will be used to instantiate the
                table's synthesizer.
        """
        return self._table_parameters.get(table_name, {})

    def get_parameters(self, table_name):
        """Return the parameters used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters used to instantiate the table's
                synthesizer.
        """
        return self._table_synthesizers.get(table_name).get_parameters()

    def update_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=self.metadata._tables[table_name],
            **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def get_metadata(self):
        """Return the ``MultiTableMetadata`` for this synthesizer."""
        return self.metadata

    def _validate_foreign_keys(self, data):
        error_msg = None
        errors = []
        for relation in self.metadata._relationships:
            child_table = data.get(relation['child_table_name'])
            parent_table = data.get(relation['parent_table_name'])
            if isinstance(child_table, pd.DataFrame) and isinstance(parent_table, pd.DataFrame):
                child_column = child_table[relation['child_foreign_key']]
                parent_column = parent_table[relation['parent_primary_key']]
                missing_values = child_column[~child_column.isin(parent_column)].unique()
                if any(missing_values):
                    message = ', '.join(missing_values[:5].astype(str))
                    if len(missing_values) > 5:
                        message = f'({message}, + more)'
                    else:
                        message = f'({message})'

                    errors.append(
                        f"Error: foreign key column '{relation['child_foreign_key']}' contains "
                        f'unknown references: {message}. All the values in this column must '
                        'reference a primary key.'
                    )
            if errors:
                error_msg = 'Relationships:\n'
                error_msg += '\n'.join(errors)

        return error_msg

    def validate(self, data):
        """Validate data.

        Args:
            data (dict):
                A dictionary with key as table name and ``pandas.DataFrame`` as value to validate.

        Raises:
            ValueError:
                Raised when data is not of type pd.DataFrame.
            InvalidDataError:
                Raised if:
                    * foreign key does not belong to a primay key
                    * data columns don't match metadata
                    * keys have missing values
                    * primary or alternate keys are not unique
                    * context columns vary for a sequence key
                    * values of a column don't satisfy their sdtype
        """
        errors = []
        missing_tables = set(self.metadata._tables) - set(data)
        if missing_tables:
            errors.append(f'The provided data is missing the tables {missing_tables}.')

        for table_name, table_data in data.items():
            try:
                self._table_synthesizers[table_name].validate(table_data)

            except InvalidDataError as error:
                error_msg = f"Table: '{table_name}'"
                for _error in error.errors:
                    error_msg += f'\nError: {_error}'

                errors.append(error_msg)

            except ValueError as error:
                errors.append(str(error))

            except KeyError:
                continue

        foreign_key_errors = self._validate_foreign_keys(data)
        if foreign_key_errors:
            errors.append(foreign_key_errors)

        if errors:
            raise InvalidDataError(errors)

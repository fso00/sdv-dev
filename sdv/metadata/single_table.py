"""Single Table Metadata."""

import json
import logging
import re
import warnings
from copy import deepcopy
from datetime import datetime

from sdv.metadata.anonymization import SDTYPE_ANONYMIZERS, is_faker_function
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.utils import read_json, validate_file_does_not_exist
from sdv.utils import cast_to_iterable, load_data_from_csv

LOGGER = logging.getLogger(__name__)


class SingleTableMetadata:
    """Single Table Metadata class."""

    _SDTYPE_KWARGS = {
        'numerical': frozenset(['computer_representation']),
        'datetime': frozenset(['datetime_format']),
        'categorical': frozenset(['order', 'order_by']),
        'boolean': frozenset([]),
        'text': frozenset(['regex_format']),
    }

    _DTYPES_TO_SDTYPES = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }

    _NUMERICAL_REPRESENTATIONS = frozenset([
        'Float', 'Int64', 'Int32', 'Int16', 'Int8',
        'UInt64', 'UInt32', 'UInt16', 'UInt8',
    ])
    _KEYS = frozenset([
        'columns',
        'primary_key',
        'alternate_keys',
        'sequence_key',
        'sequence_index',
        'METADATA_SPEC_VERSION'
    ])
    METADATA_SPEC_VERSION = 'SINGLE_TABLE_V1'

    def _validate_numerical(self, column_name, **kwargs):
        representation = kwargs.get('computer_representation')
        if representation and representation not in self._NUMERICAL_REPRESENTATIONS:
            raise InvalidMetadataError(
                f"Invalid value for 'computer_representation' '{representation}'"
                f" for column '{column_name}'."
            )

    @staticmethod
    def _validate_datetime(column_name, **kwargs):
        datetime_format = kwargs.get('datetime_format')
        if datetime_format is not None:
            try:
                formated_date = datetime.now().strftime(datetime_format)
            except Exception as exception:
                raise InvalidMetadataError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                ) from exception

            matches = re.findall('(%.)|(%)', formated_date)
            if matches:
                raise InvalidMetadataError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                )

    @staticmethod
    def _validate_categorical(column_name, **kwargs):
        order = kwargs.get('order')
        order_by = kwargs.get('order_by')
        if order is not None and order_by is not None:
            raise InvalidMetadataError(
                f"Categorical column '{column_name}' has both an 'order' and 'order_by' "
                'attribute. Only 1 is allowed.'
            )
        if order_by is not None and order_by not in ('numerical_value', 'alphabetical'):
            raise InvalidMetadataError(
                f"Unknown ordering method '{order_by}' provided for categorical column "
                f"'{column_name}'. Ordering method must be 'numerical_value' or 'alphabetical'."
            )
        if (isinstance(order, list) and not len(order)) or\
           (not isinstance(order, list) and order is not None):
            raise InvalidMetadataError(
                f"Invalid order value provided for categorical column '{column_name}'. "
                "The 'order' must be a list with 1 or more elements."
            )

    @staticmethod
    def _validate_text(column_name, **kwargs):
        regex = kwargs.get('regex_format', '')
        try:
            re.compile(regex)
        except Exception as exception:
            raise InvalidMetadataError(
                f"Invalid regex format string '{regex}' for text column '{column_name}'."
            ) from exception

    @staticmethod
    def _validate_pii(column_name, **kwargs):
        pii_value = kwargs['pii']
        if not isinstance(pii_value, bool):
            raise InvalidMetadataError(
                f"Parameter 'pii' is set to an invalid attribute ('{pii_value}') for column "
                f"'{column_name}'. Expected a value of True or False."
            )

    def __init__(self):
        self.columns = {}
        self.primary_key = None
        self._alternate_keys = []
        self._sequence_key = None
        self._sequence_index = None
        self._version = self.METADATA_SPEC_VERSION

    def _validate_unexpected_kwargs(self, column_name, sdtype, **kwargs):
        expected_kwargs = self._SDTYPE_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(kwargs) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = sorted(unexpected_kwargs)
            unexpected_kwargs = ', '.join(unexpected_kwargs)
            raise InvalidMetadataError(
                f"Invalid values '({unexpected_kwargs})' for {sdtype} column '{column_name}'.")

    def _validate_sdtype(self, sdtype):
        is_default_sdtype = sdtype in self._SDTYPE_KWARGS
        is_anonymized_type = sdtype in SDTYPE_ANONYMIZERS
        if not (is_default_sdtype or is_anonymized_type or is_faker_function(sdtype)):
            raise InvalidMetadataError(
                f"Invalid sdtype : '{sdtype}' is not recognized. Please use one of the "
                'supported SDV sdtypes.'
            )

    def _validate_column(self, column_name, sdtype, **kwargs):
        self._validate_sdtype(sdtype)
        self._validate_unexpected_kwargs(column_name, sdtype, **kwargs)
        if sdtype == 'categorical':
            self._validate_categorical(column_name, **kwargs)
        elif sdtype == 'numerical':
            self._validate_numerical(column_name, **kwargs)
        elif sdtype == 'datetime':
            self._validate_datetime(column_name, **kwargs)
        elif sdtype == 'text':
            self._validate_text(column_name, **kwargs)
        elif 'pii' in kwargs:
            self._validate_pii(column_name, **kwargs)

    def add_column(self, column_name, **kwargs):
        """Add a column to the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be added.
            kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``InvalidMetadataError`` if the column already exists.
            - ``InvalidMetadataError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              given ``sdtype``.
            - ``InvalidMetadataError`` if the ``pii`` value is not ``True`` or ``False`` when
               present.
        """
        if column_name in self.columns:
            raise InvalidMetadataError(
                f"Column name '{column_name}' already exists. Use 'update_column' "
                'to update an existing column.'
            )

        sdtype = kwargs.get('sdtype')
        if sdtype is None:
            raise InvalidMetadataError(f"Please provide a 'sdtype' for column '{column_name}'.")

        self._validate_column(column_name, **kwargs)
        column_kwargs = deepcopy(kwargs)
        if sdtype not in self._SDTYPE_KWARGS:
            pii = column_kwargs.get('pii', True)
            column_kwargs['pii'] = pii

        self.columns[column_name] = column_kwargs

    def _validate_column_exists(self, column_name):
        if column_name not in self.columns:
            raise InvalidMetadataError(
                f"Column name ('{column_name}') does not exist in the table. "
                "Use 'add_column' to add new column."
            )

    def update_column(self, column_name, **kwargs):
        """Update an existing column in the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``InvalidMetadataError`` if the column doesn't already exist in the
              ``SingleTableMetadata``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              current
              ``sdtype``.
            - ``InvalidMetadataError`` if the ``pii`` value is not ``True`` or ``False`` when
               present.
        """
        self._validate_column_exists(column_name)
        _kwargs = deepcopy(kwargs)
        if 'sdtype' in kwargs:
            sdtype = kwargs.pop('sdtype')
        else:
            sdtype = self.columns[column_name]['sdtype']
            _kwargs['sdtype'] = sdtype

        self._validate_column(column_name, sdtype, **kwargs)
        self.columns[column_name] = _kwargs

    def to_dict(self):
        """Return a python ``dict`` representation of the ``SingleTableMetadata``."""
        metadata = {}
        for key in self._KEYS:
            if key in ('columns', 'primary_key'):
                value = getattr(self, f'{key}')
            else:
                value = \
                    getattr(self, f'_{key}') if key != 'METADATA_SPEC_VERSION' else self._version
            if value:
                metadata[key] = value

        return deepcopy(metadata)

    def _detect_columns(self, data):
        for field in data:
            clean_data = data[field].dropna()
            kind = clean_data.infer_objects().dtype.kind
            self.columns[field] = {'sdtype': self._DTYPES_TO_SDTYPES.get(kind, 'categorical')}

    def detect_from_dataframe(self, data):
        """Detect the metadata from a ``pd.DataFrame`` object.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.

        Args:
            data (pandas.DataFrame):
                ``pandas.DataFrame`` to detect the metadata from.
        """
        if self.columns:
            raise InvalidMetadataError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        self._detect_columns(data)

        LOGGER.info('Detected metadata:')
        LOGGER.info(json.dumps(self.to_dict(), indent=4))

    def detect_from_csv(self, filepath, pandas_kwargs=None):
        """Detect the metadata from a ``csv`` file.

        This method automatically detects the ``sdtypes`` for a given ``csv`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``csv`` file.
            pandas_kwargs (dict):
                A python dictionary of with string and value accepted by ``pandas.read_csv``
                function. Defaults to ``None``.
        """
        if self.columns:
            raise InvalidMetadataError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        data = load_data_from_csv(filepath, pandas_kwargs)
        self.detect_from_dataframe(data)

    @staticmethod
    def _validate_datatype(column_name):
        """Check whether column_name is a string or a tuple of strings."""
        return isinstance(column_name, str) or \
            isinstance(column_name, tuple) and all(isinstance(i, str) for i in column_name)

    def _validate_keys_sdtype(self, keys, key_type):
        """Validate that no key is of type 'categorical'."""
        bad_sdtypes = ('boolean', 'categorical')
        categorical_keys = sorted(
            {key for key in keys if self.columns[key]['sdtype'] in bad_sdtypes}
        )
        if categorical_keys:
            raise InvalidMetadataError(
                f"The {key_type}_keys {categorical_keys} cannot be type 'categorical' or "
                "'boolean'."
            )

    def _validate_key(self, column_name, key_type):
        """Validate the primary and sequence keys."""
        if column_name is not None:
            if not self._validate_datatype(column_name):
                raise InvalidMetadataError(
                    f"'{key_type}_key' must be a string or tuple of strings.")

            keys = {column_name} if isinstance(column_name, str) else set(column_name)
            invalid_ids = keys - set(self.columns)
            if invalid_ids:
                raise InvalidMetadataError(
                    f'Unknown {key_type} key values {invalid_ids}.'
                    ' Keys should be columns that exist in the table.'
                )

            self._validate_keys_sdtype(keys, key_type)

    def set_primary_key(self, column_name):
        """Set the metadata primary key.

        Args:
            column_name (str, tuple):
                Name (or tuple of names) of the primary key column(s).
        """
        self._validate_key(column_name, 'primary')
        if column_name in self._alternate_keys:
            warnings.warn(
                f'{column_name} is currently set as an alternate key and will be removed from '
                'that list.'
            )
            self._alternate_keys.remove(column_name)

        if self.primary_key is not None:
            warnings.warn(
                f'There is an existing primary key {self.primary_key}.'
                ' This key will be removed.'
            )

        self.primary_key = column_name

    def set_sequence_key(self, column_name):
        """Set the metadata sequence key.

        Args:
            column_name (str, tuple):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_key(column_name, 'sequence')
        if self._sequence_key is not None:
            warnings.warn(
                f'There is an existing sequence key {self._sequence_key}.'
                ' This key will be removed.'
            )

        self._sequence_key = column_name

    def _validate_alternate_keys(self, column_names):
        if not isinstance(column_names, list) or \
           not all(self._validate_datatype(column_name) for column_name in column_names):
            raise InvalidMetadataError(
                "'alternate_keys' must be a list of strings or a list of tuples of strings."
            )

        keys = set()
        for column_name in column_names:
            keys.update({column_name} if isinstance(column_name, str) else set(column_name))

        invalid_ids = keys - set(self.columns)
        if invalid_ids:
            raise InvalidMetadataError(
                f'Unknown alternate key values {invalid_ids}.'
                ' Keys should be columns that exist in the table.'
            )

        if self.primary_key in column_names:
            raise InvalidMetadataError(
                f"Invalid alternate key '{self.primary_key}'. The key is "
                'already specified as a primary key.'
            )

        self._validate_keys_sdtype(keys, 'alternate')

    def add_alternate_keys(self, column_names):
        """Set the metadata alternate keys.

        Args:
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self._validate_alternate_keys(column_names)
        for column in column_names:
            if column in self._alternate_keys:
                warnings.warn(f'{column} is already an alternate key.')
            else:
                self._alternate_keys.append(column)

    def _validate_sequence_index(self, column_name):
        if not isinstance(column_name, str):
            raise InvalidMetadataError("'sequence_index' must be a string.")

        if column_name not in self.columns:
            column_name = {column_name}
            raise InvalidMetadataError(
                f'Unknown sequence index value {column_name}.'
                ' Keys should be columns that exist in the table.'
            )

        sdtype = self.columns[column_name].get('sdtype')
        if sdtype not in ['datetime', 'numerical']:
            raise InvalidMetadataError(
                "The sequence_index must be of type 'datetime' or 'numerical'.")

    def set_sequence_index(self, column_name):
        """Set the metadata sequence index.

        Args:
            column_name (str):
                Name of the sequence index column.
        """
        self._validate_sequence_index(column_name)
        self._sequence_index = column_name

    def _validate_sequence_index_not_in_sequence_key(self):
        """Check that ``_sequence_index`` and ``_sequence_key`` don't overlap."""
        seq_key = self._sequence_key
        sequence_key = set(cast_to_iterable(seq_key))
        if self._sequence_index in sequence_key or seq_key is None:
            index = {self._sequence_index}
            raise InvalidMetadataError(
                f"'sequence_index' and 'sequence_key' have the same value {index}."
                ' These columns must be different.'
            )

    def _append_error(self, errors, method, *args, **kwargs):
        """Inplace, append the produced error to the passed ``errors`` list."""
        try:
            method(*args, **kwargs)
        except InvalidMetadataError as e:
            errors.append(e)

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        errors = []
        # Validate keys
        self._append_error(errors, self._validate_key, self.primary_key, 'primary')
        self._append_error(errors, self._validate_key, self._sequence_key, 'sequence')
        if self._sequence_index:
            self._append_error(errors, self._validate_sequence_index, self._sequence_index)
            self._append_error(errors, self._validate_sequence_index_not_in_sequence_key)

        self._append_error(errors, self._validate_alternate_keys, self._alternate_keys)

        # Validate columns
        for column, kwargs in self.columns.items():
            self._append_error(errors, self._validate_column, column, **kwargs)

        if errors:
            raise InvalidMetadataError(
                'The following errors were found in the metadata:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

    def save_to_json(self, filepath):
        """Save the current ``SingleTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file to be written.

        Raises:
            Raises an ``Error`` if the path already exists.
        """
        validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        metadata['METADATA_SPEC_VERSION'] = self.METADATA_SPEC_VERSION
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

    @classmethod
    def _load_from_dict(cls, metadata):
        """Create a ``SingleTableMetadata`` instance from a python ``dict``.

        Args:
            metadata (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.

        Returns:
            Instance of ``SingleTableMetadata``.
        """
        instance = cls()
        for key in instance._KEYS:
            value = deepcopy(metadata.get(key))
            if key in ('columns', 'primary_key'):
                setattr(instance, f'{key}', value)
            elif value:
                setattr(instance, f'_{key}', value)

        return instance

    @classmethod
    def load_from_json(cls, filepath):
        """Create an instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``METADATA_SPEC_VERSION``.

        Returns:
            A ``SingleTableMetadata`` instance.
        """
        metadata = read_json(filepath)
        if 'METADATA_SPEC_VERSION' not in metadata:
            raise InvalidMetadataError(
                'This metadata file is incompatible with the ``SingleTableMetadata`` '
                'class and version.'
            )

        return cls._load_from_dict(metadata)

    def __repr__(self):
        """Pretty print the ``SingleTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    @classmethod
    def upgrade_metadata(cls, old_filepath, new_filepath):
        """Upgrade an old metadata file to the ``V1`` schema.

        Args:
            old_filepath (str):
                String that represents the ``path`` to the old metadata ``json`` file.
            new_file_path (str):
                String that represents the ``path`` to save the upgraded metadata to.

        Raises:
            - ``ValueError`` if the path already exists.
            - ``InvalidMetadataError`` if multiple tables are specified in the JSON.
        """
        validate_file_does_not_exist(new_filepath)
        old_metadata = read_json(old_filepath)
        if 'tables' in old_metadata:
            tables = old_metadata.get('tables')
            if len(tables) > 1:
                raise InvalidMetadataError(
                    'There are multiple tables specified in the JSON. '
                    'Try using the MultiTableMetadata class to upgrade this file.'
                )

            else:
                old_metadata = list(tables.values())[0]

        new_metadata = convert_metadata(old_metadata)
        metadata = cls._load_from_dict(new_metadata)
        metadata.save_to_json(new_filepath)

        try:
            metadata.validate()
        except InvalidMetadataError as error:
            message = (
                'Successfully converted the old metadata, but the metadata was not valid. '
                f'To use this with the SDV, please fix the following errors.\n {str(error)}'
            )
            warnings.warn(message)

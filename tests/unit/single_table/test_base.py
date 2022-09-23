from unittest.mock import Mock, patch

import pytest

from sdv.single_table.base import BaseSynthesizer
from sdv.metadata.single_table import SingleTableMetadata
import pandas as pd
import pytest
import numpy as np


class TestBaseSynthesizer:

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__(self, mock_data_processor):
        """Test instantiating with default values."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__custom(self, mock_data_processor):
        """Test that instantiating with custom parameters are properly stored in the instance."""
        # Setup
        metadata = Mock()

        # Run
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance._data_processor == mock_data_processor.return_value
        mock_data_processor.assert_called_once_with(metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_parameters(self, mock_data_processor):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {'enforce_min_max_values': False, 'enforce_rounding': False}

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_metadata(self, mock_data_processor):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = Mock()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata

    @patch('sdv.single_table.base.DataProcessor')
    def test__fit(self, mock_data_processor):
        """Test that ``NotImplementedError`` is being raised."""
        # Setup
        metadata = Mock()
        data = Mock()
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._fit(data)

    def test_fit_processed_data(self):
        """Test that ``fit_processed_data`` calls the ``_fit``."""
        # Setup
        instance = Mock()
        processed_data = Mock()

        # Run
        BaseSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance._fit.assert_called_once_with(processed_data)

    def test_fit(self):
        """Test that ``fit`` calls ``preprocess`` and the ``fit_processed_data``.

        When fitting, the synthsizer has to ``preprocess`` the data and with the output
        of this method, call the ``fit_processed_data``
        """
        # Setup
        instance = Mock()
        processed_data = Mock()

        # Run
        BaseSynthesizer.fit(instance, processed_data)

        # Assert
        instance.preprocess.assert_called_once_with(processed_data)
        instance.fit_processed_data.assert_called_once_with(instance.preprocess.return_value)
    
    def test_validate_keys(self):
        data = pd.DataFrame({
            'pk_col': [0,1,2,3,4,5],
            'sk_col': [0,1,2,3,4,5],
            'ak_col': [0,1,2,3,4,5],

        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col', sdtype='numerical')
        metadata.add_column('ak_col', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        metadata.set_alternate_keys(['ak_col'])
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)

    def test_validate_keys_with_missing_values(self):
        data = pd.DataFrame({
            'pk_col': [0,1,2,3,4,np.nan],
            'sk_col': [0,1,2,3,4,np.nan],
            'ak_col': [0,1,2,3,4,np.nan],

        })
        metadata = SingleTableMetadata()
        metadata.add_column('pk_col', sdtype='numerical')
        metadata.add_column('sk_col', sdtype='numerical')
        metadata.add_column('ak_col', sdtype='numerical')
        metadata.set_primary_key('pk_col')
        metadata.set_sequence_key('sk_col')
        metadata.set_alternate_keys(['ak_col'])
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        instance.validate(data)
        

    
    def test_validate(self):
        """Test it doesn't crash."""
        # Setup
        data = pd.DataFrame({
            'numerical_col': [np.nan, None, float('nan'), -1, 0, 1.54],
            'date_col': [np.nan, None, float('nan'), '2021-02-10', '2021-05-10', '2021-08-11'],
            'bool_col': [np.nan, None, float('nan'), True, False, True],

        })
        metadata = SingleTableMetadata()
        metadata.add_column('numerical_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='datetime')
        metadata.add_column('bool_col', sdtype='boolean')
        instance = BaseSynthesizer(metadata)

        # Run
        instance.validate(data)
    
    def test_validate_all_together(self):
        ...
    
    def test_validate_raises(self):
        """Test it crashes with right error."""
        # Setup
        data = pd.DataFrame({
            'numerical_col': ['a', 1, '10', 5, True, 'b'],
            'date_col': ['10', 10, True, '2021-05-10', '10-10-10-10', 'Bla'],
            'bool_col': ['a', 1, '10', 5, True, 'b'],
        })
        metadata = SingleTableMetadata()
        metadata.add_column('numerical_col', sdtype='numerical')
        metadata.add_column('date_col', sdtype='datetime')
        metadata.add_column('bool_col', sdtype='boolean')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        instance.validate(data)
        
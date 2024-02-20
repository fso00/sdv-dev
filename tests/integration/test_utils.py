import re

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.metadata import MultiTableMetadata
from sdv.utils import drop_unknown_references


@pytest.fixture()
def metadata():
    return MultiTableMetadata.load_from_dict(
        {
            'tables': {
                'parent': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'A': {'sdtype': 'categorical'},
                        'B': {'sdtype': 'numerical'}
                    },
                    'primary_key': 'id'
                },
                'child': {
                    'columns': {
                        'parent_id': {'sdtype': 'id'},
                        'C': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'child_table_name': 'child',
                    'parent_primary_key': 'id',
                    'child_foreign_key': 'parent_id'
                }
            ]
        }
    )


@pytest.fixture()
def data():
    parent = pd.DataFrame(data={
        'id': [0, 1, 2, 3, 4],
        'A': [True, True, False, True, False],
        'B': [0.434, 0.312, 0.212, 0.339, 0.491]
    })

    child = pd.DataFrame(data={
        'parent_id': [0, 1, 2, 2, 5],
        'C': ['Yes', 'No', 'Maye', 'No', 'No']
    })

    return {
        'parent': parent,
        'child': child
    }


def test_drop_unknown_references(metadata, data):
    """Test ``drop_unknown_references`` end to end."""
    # Run
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        'Relationships:\n'
        "Error: foreign key column 'parent_id' contains unknown references: (5)"
        '. All the values in this column must reference a primary key.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        metadata.validate_data(data)

    cleaned_data = drop_unknown_references(metadata, data)
    metadata.validate_data(cleaned_data)

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert len(cleaned_data['child']) == 4


def test_drop_unknown_references_drop_missing_values(metadata, data):
    """Test ``drop_unknown_references`` when there is missing values in the foreign keys."""
    # Setup
    data['child'].loc[3, 'parent_id'] = np.nan

    # Run
    cleaned_data = drop_unknown_references(metadata, data)
    metadata.validate_data(cleaned_data)

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:3])
    assert len(cleaned_data['child']) == 3


def test_drop_unknown_references_not_drop_missing_values(metadata, data):
    """Test ``drop_unknown_references`` when the missing values in the foreign keys are kept."""
    # Setup
    data['child'].loc[3, 'parent_id'] = np.nan

    # Run
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        'Relationships:\n'
        "Error: foreign key column 'parent_id' contains unknown references: (nan)"
        '. All the values in this column must reference a primary key.'
    )

    cleaned_data = drop_unknown_references(metadata, data, drop_missing_values=False)
    with pytest.raises(InvalidDataError, match=expected_message):
        metadata.validate_data(cleaned_data)

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert len(cleaned_data['child']) == 4

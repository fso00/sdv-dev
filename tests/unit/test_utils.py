"""Tests for the sdv.utils module."""

from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import pkg_resources

from sdv.tabular import GaussianCopula
from sdv.utils import (cast_to_iterable, get_package_versions, load_data_from_csv,
    throw_version_mismatch_warning)


def test_cast_to_iterable():
    """Test ``cast_to_iterable``.

    Test that ``cast_to_iterable`` converts a signle object into a ``list`` but does not convert
    a ``list`` into a list inside a list.
    """
    # Setup
    value = 'abc'
    list_value = ['ab']

    # Run
    value = cast_to_iterable(value)
    list_value = cast_to_iterable(list_value)

    # Assert
    assert value == ['abc']
    assert list_value == ['ab']


@patch('sdv.utils.pkg_resources.get_distribution')
def test_get_package_versions_no_model(get_distribution_mock):
    """Test the ``get_package_versions`` method when there is no model.

    Expect that the package versions for `sdv` and `rdt` are returned.

    Setup:
        - Patch the versions of sdv and rdt
    Input:
        - model is None
    Output:
        - a dict mapping libraries to library versions.
    """
    # Setup
    expected = {
        'sdv': get_distribution_mock.return_value.version,
        'rdt': get_distribution_mock.return_value.version,
    }

    # Run
    versions = get_package_versions(model=None)

    # Assert
    assert get_distribution_mock.has_calls([call('sdv'), call('rdt')])
    assert versions == expected


@patch('sdv.utils.pkg_resources.get_distribution')
def test_get_package_versions_valid_model(get_distribution_mock):
    """Test the ``get_package_versions`` method with model.

    Expect that the package versions for `sdv` and `rdt` are returned.

    Setup:
        - Patch the versions of sdv and rdt
        - Create a model named GaussianCopula
    Input:
        - model
    Output:
        - a dict mapping libraries to library versions.
    """
    # Setup
    expected = {
        'sdv': get_distribution_mock.return_value.version,
        'rdt': get_distribution_mock.return_value.version,
        'copulas': get_distribution_mock.return_value.version,
    }
    model = Mock(spec=GaussianCopula)

    # Run
    versions = get_package_versions(model=model)

    # Assert
    assert get_distribution_mock.has_calls([call('sdv'), call('rdt'), call('copulas')])
    assert versions == expected


@patch('sdv.utils.pkg_resources.get_distribution')
def test_get_package_versions_error(get_distribution_mock):
    """Test the ``get_package_versions`` method with model.

    Expect that the package versions for `sdv` and `rdt` are returned, and that the unavailable
    packages are ignored.

    Setup:
        - Patch the versions of sdv and rdt
        - Create a model named GaussianCopula
    Input:
        - model
    Output:
        - a dict mapping libraries to library versions.
    """
    # Setup
    dist_mock = Mock()
    get_distribution_mock.side_effect = [
        dist_mock, pkg_resources.ResolutionError(), dist_mock
    ]

    expected = {
        'sdv': dist_mock.version,
        'copulas': dist_mock.version,
    }
    model = Mock(spec=GaussianCopula)

    # Run
    versions = get_package_versions(model=model)

    # Assert
    assert get_distribution_mock.has_calls([call('sdv'), call('rdt'), call('copulas')])
    assert versions == expected


@patch('sdv.utils.pkg_resources.warnings')
@patch('sdv.utils.pkg_resources.get_distribution')
def test_throw_version_mismatch_warning_no_mismatches(get_distribution_mock, warnings_mock):
    """Test the ``generate_version_mismatch_warning`` method with no mismatched versions.

    Expect that None is returned.

    Setup:
        - Mock ``pkg_resources.get_distribution`` to return the same versions.
    Input:
        - package versions mapping
    Output:
        - None
    """
    # Setup
    dist_mock = Mock()
    get_distribution_mock.side_effect = [dist_mock, dist_mock]
    package_versions = {'sdv': dist_mock.version, 'rdt': dist_mock.version}

    # Run
    throw_version_mismatch_warning(package_versions)

    # Assert
    assert get_distribution_mock.hass_calls([call('sdv'), call('rdt')])
    assert warnings_mock.warn.call_count == 0


@patch('sdv.utils.pkg_resources.warnings')
@patch('sdv.utils.pkg_resources.get_distribution')
def test_generate_version_mismatch_warning_mismatched_version(get_distribution_mock,
                                                              warnings_mock):
    """Test the ``generate_version_mismatch_warning`` method with one mismatched version.

    Expect that the proper warning is thrown.

    Setup:
        - Mock ``pkg_resources.get_distribution`` to return the same versions, except for one.
    Input:
        - package versions mapping
    Side Effect:
        - warning
    """
    # Setup
    dist_mock = Mock()
    get_distribution_mock.side_effect = [dist_mock, dist_mock]
    package_versions = {'sdv': dist_mock.version, 'rdt': 'rdt_version'}
    expected_str = ('The libraries used to create the model have older versions than your '
                    'current setup. This may cause errors when sampling.\nrdt used version '
                    '`rdt_version`; current version is `{0}`'.format(dist_mock.version))

    # Run
    throw_version_mismatch_warning(package_versions)

    # Assert
    assert get_distribution_mock.hass_calls([call('sdv'), call('rdt')])
    assert warnings_mock.warn.called_once_with(expected_str)


@patch('sdv.utils.pkg_resources.warnings')
@patch('sdv.utils.pkg_resources.get_distribution')
def test_generate_version_mismatch_warning_mismatched_version_error(get_distribution_mock,
                                                                    warnings_mock):
    """Test the ``generate_version_mismatch_warning`` method with one mismatched version.

    Expect that if we error out when trying to get the current package version for one
    package, we treat that as a mismatch and the proper warning is thrown.

    Setup:
        - Mock ``pkg_resources.get_distribution`` to return the same versions, except for one,
          which errors out.
    Input:
        - package versions mapping
    Side Effect:
        - warning
    """
    # Setup
    dist_mock = Mock()
    get_distribution_mock.side_effect = [dist_mock, pkg_resources.ResolutionError()]
    package_versions = {'sdv': dist_mock.version, 'rdt': 'rdt_version'}
    expected_str = ('The libraries used to create the model have older versions than your '
                    'current setup. This may cause errors when sampling.\nrdt used version '
                    '`rdt_version`; current version is ``')

    # Run
    throw_version_mismatch_warning(package_versions)

    # Assert
    assert get_distribution_mock.hass_calls([call('sdv'), call('rdt')])
    assert warnings_mock.warn.called_once_with(expected_str)


@patch('sdv.utils.pkg_resources.warnings')
def test_generate_version_mismatch_warning_no_package_versions(warnings_mock):
    """Test the ``generate_version_mismatch_warning`` method with None as input.

    Expect that if the model didn't have a `_package_versions` attribute, we throw
    a generic warning.

    Input:
        - package versions is None
    Side Effect:
        - Generic warning is thrown.
    """
    # Setup
    package_versions = None
    expected_str = ('The libraries used to create the model have older versions than your '
                    'current setup. This may cause errors when sampling.')

    # Run
    throw_version_mismatch_warning(package_versions)

    # Assert
    assert warnings_mock.warn.called_once_with(expected_str)


@patch('sdv.utils.pd.read_csv')
def test_load_data_from_csv(read_csv_mock):
    """Test that the data is loaded from the path.

    This function should create a filepath from the provided string and pass along any key-word
    args to ``pandas.read_csv`` in order to load the data.
    """
    # Setup
    expected_data = pd.DataFrame({'number': [1., 2., 3.], 'text': ['1', '2', '3']})
    read_csv_mock.return_value = expected_data

    # Run
    loaded_data = load_data_from_csv('data.csv', pandas_kwargs={'dtype': {'number': 'float'}})

    # Assert
    pd.testing.assert_frame_equal(loaded_data, expected_data)
    read_csv_mock.assert_called_once_with(Path('data.csv'), dtype={'number': 'float'})

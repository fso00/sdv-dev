"""Tests for the sdv.constraints.base module."""
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from copulas.univariate import GaussianUnivariate

from sdv.constraints.base import (
    Constraint, ColumnsModel, _get_qualified_name, get_subclasses, import_object)
from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import ColumnFormula, FixedCombinations
from sdv.errors import ConstraintsNotMetError


def test__get_qualified_name_class():
    """Test the ``_get_qualified_name`` function, if a class is passed.

    The ``_get_qualified_name`` function is expected to:
    - Return the Fully Qualified Name from a class.

    Input:
    - A class.
    Output:
    - The class qualified name.
    """
    # Run
    fully_qualified_name = _get_qualified_name(Constraint)

    # Assert
    expected_name = 'sdv.constraints.base.Constraint'
    assert fully_qualified_name == expected_name


def test__get_qualified_name_function():
    """Test the ``_get_qualified_name`` function, if a function is passed.

    The ``_get_qualified_name`` function is expected to:
    - Return the Fully Qualified Name from a function.

    Input:
    - A function.
    Output:
    - The function qualified name.
    """
    # Run
    fully_qualified_name = _get_qualified_name(_get_qualified_name)

    # Assert
    expected_name = 'sdv.constraints.base._get_qualified_name'
    assert fully_qualified_name == expected_name


def test_get_subclasses():
    """Test the ``get_subclasses`` function.

    The ``get_subclasses`` function is expected to:
    - Recursively find subclasses for the class object passed.

    Setup:
    - Create three classes, Parent, Child and GrandChild,
      which inherit of each other hierarchically.

    Input:
    - The Parent class.
    Output:
    - Dict of the subclasses of the class: ``Child`` and ``GrandChild`` classes.
    """
    # Setup
    class Parent:
        pass

    class Child(Parent):
        pass

    class GrandChild(Child):
        pass

    # Run
    subclasses = get_subclasses(Parent)

    # Assert
    expected_subclasses = {
        'Child': Child,
        'GrandChild': GrandChild
    }

    assert subclasses == expected_subclasses


def test_import_object_class():
    """Test the ``import_object`` function, when importing a class.

    The ``import_object`` function is expected to:
    - Import a class from its qualifed name.

    Input:
    - Qualified name of the class.
    Output:
    - The imported class.
    """
    # Run
    obj = import_object('sdv.constraints.base.Constraint')

    # Assert
    assert obj is Constraint


def test_import_object_function():
    """Test the ``import_object`` function, when importing a function.

    The ``import_object`` function is expected to:
    - Import a function from its qualifed name.

    Input:
    - Qualified name of the function.
    Output:
    - The imported function.
    """
    # Run
    imported = import_object('sdv.constraints.base.import_object')

    # Assert
    assert imported is import_object


class TestConstraint():

    def test__identity(self):
        """Test ```Constraint._identity`` method.

        ``_identity`` method should return whatever it is passed.

        Input:
            - anything
        Output:
            - Input
        """
        # Run
        instance = Constraint('all')
        output = instance._identity('input')

        # Asserts
        assert output == 'input'

    def test___init___transform(self):
        """Test ```Constraint.__init__`` method when 'transform' is passed.

        If 'transform' is given, the ``__init__`` method should replace the ``is_valid`` method
        with an identity and leave ``transform`` and ``reverse_transform`` untouched.

        Input:
            - transform
        Side effects:
            - is_valid == identity
            - transform != identity
            - reverse_transform != identity
        """
        # Run
        instance = Constraint(handling_strategy='transform')

        # Asserts
        assert instance.filter_valid == instance._identity
        assert instance.transform != instance._identity
        assert instance.reverse_transform != instance._identity

    def test___init___reject_sampling(self):
        """Test ``Constraint.__init__`` method when 'reject_sampling' is passed.

        If 'reject_sampling' is given, the ``__init__`` method should replace the ``transform``
        and ``reverse_transform`` methods with an identity and leave ``is_valid`` untouched.

        Input:
            - reject_sampling
        Side effects:
            - is_valid != identity
            - transform == identity
            - reverse_transform == identity
        """
        # Run
        instance = Constraint(handling_strategy='reject_sampling')

        # Asserts
        assert instance.filter_valid != instance._identity
        assert instance.transform == instance._identity_with_validation
        assert instance.reverse_transform == instance._identity

    def test___init___all(self):
        """Test ``Constraint.__init__`` method when 'all' is passed.

        If 'all' is given, the ``__init__`` method should leave ``transform``,
        ``reverse_transform`` and ``is_valid`` untouched.

        Input:
            - all
        Side effects:
            - is_valid != identity
            - transform != identity
            - reverse_transform != identity
        """
        # Run
        instance = Constraint(handling_strategy='all')

        # Asserts
        assert instance.filter_valid != instance._identity
        assert instance.transform != instance._identity
        assert instance.reverse_transform != instance._identity

    def test___init___not_kown(self):
        """Test ``Constraint.__init__`` method when a not known ``handling_strategy`` is passed.

        If a not known ``handling_strategy`` is given, a ValueError is raised.

        Input:
            - not_known
        Side effects:
            - ValueError
        """
        # Run
        with pytest.raises(ValueError):
            Constraint(handling_strategy='not_known')

    def test_fit(self):
        """Test the ``Constraint.fit`` method.

        The base ``Constraint.fit`` method is expected to:
        - Call ``_fit`` method.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        instance = Constraint(handling_strategy='transform')
        instance._fit = Mock()

        # Run
        instance.fit(table_data)

        # Assert
        instance._fit.assert_called_once_with(table_data)

    def test__validate_data_on_constraints(self):
        """Test the ``_validate_data_on_constraint`` method.

        Expect that the method calls ``is_valid`` when the constraint columns
        are in the given data.

        Input:
        - Table data
        Output:
        - None
        Side Effects:
        - No error
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint = Constraint(handling_strategy='transform')
        constraint.constraint_columns = ['a', 'b']
        constraint.is_valid = Mock()

        # Run
        constraint._validate_data_on_constraint(data)

        # Assert
        constraint.is_valid.assert_called_once_with(data)

    def test__validate_data_on_constraints_invalid_input(self):
        """Test the ``_validate_data_on_constraint`` method.

        Expect that the method raises an error when the constraint columns
        are in the given data and the ``is_valid`` returns False for any row.

        Input:
        - Table data contains an invalid row
        Output:
        - None
        Side Effects:
        - A ``ConstraintsNotMetError`` is thrown
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint = Constraint(handling_strategy='transform')
        constraint.constraint_columns = ['a', 'b']
        constraint.is_valid = Mock(return_value=pd.Series([True, False, True]))

        # Run / Assert
        with pytest.raises(ConstraintsNotMetError):
            constraint._validate_data_on_constraint(data)

    def test__validate_data_on_constraints_missing_cols(self):
        """Test the ``_validate_data_on_constraint`` method.

        Expect that the method doesn't do anything when the columns are not in the given data.

        Input:
        - Table data that is missing a constraint column
        Output:
        - None
        Side Effects:
        - No error
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint = Constraint(handling_strategy='transform')
        constraint.constraint_columns = ['a', 'b', 'c']
        constraint.is_valid = Mock()

        # Run
        constraint._validate_data_on_constraint(data)

        # Assert
        assert not constraint.is_valid.called

    def test_transform(self):
        """Test the ``Constraint.transform`` method.

        When no constraints are passed, it behaves like an identity method,
        to be optionally overwritten by subclasses.

        The ``Constraint.transform`` method is expected to:
        - Return the input data unmodified.
        Input:
        - a DataFrame
        Output:
        - Input
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        output = instance.transform(pd.DataFrame({'col': ['input']}))

        # Assert
        pd.testing.assert_frame_equal(output, pd.DataFrame({'col': ['input']}))

    def test_transform_calls__transform(self):
        """Test that the ``Constraint.transform`` method calls ``_transform``.

        The ``Constraint.transform`` method is expected to:
        - Return value returned by ``_transform``.

        Input:
        - Anything
        Output:
        - Result of ``_transform(input)``
        """
        # Setup
        constraint_mock = Mock()
        constraint_mock._transform.return_value = 'the_transformed_data'
        constraint_mock._validate_columns.return_value = pd.DataFrame()

        # Run
        output = Constraint.transform(constraint_mock, 'input')

        # Assert
        assert output == 'the_transformed_data'

    def test_transform_model_disabled_any_columns_missing(self):
        """Test the ``Constraint.transform`` method with invalid data.

        If ``table_data`` is missing any columns it should raise a
        ``MissingConstraintColumnError``.

        The ``Constraint.transform`` method is expected to:
        - Raise ``MissingConstraintColumnError``.
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a',)

        # Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame([[1, 2], [3, 4]], columns=['b', 'c']))

    def test_transform_model_enabled_all_columns_missing(self):
        """Test the ``Constraint.transform`` method with missing columns.

        If ``table_data`` is missing all of the ``constraint_columns`` a
        ``MissingConstraintColumnError`` is raised.

        The ``Constraint.transform`` method is expected to:
        - Raise ``MissingConstraintColumnError``.
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a',)

        # Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame())

    def test_fit_transform(self):
        """Test the ``Constraint.fit_transform`` method.

        The ``Constraint.fit_transform`` method is expected to:
        - Call the ``fit`` method.
        - Call the ``transform`` method.
        - Return the input data unmodified.

        Input:
        - Anything
        Output:
        - self.transform output
        Side Effects:
        - self.fit is called with input
        - self.transform is called with input
        """
        # Setup
        constraint_mock = Mock()
        constraint_mock.transform.return_value = 'the_transformed_data'

        # Run
        data = 'my_data'
        output = Constraint.fit_transform(constraint_mock, data)

        # Assert
        assert output == 'the_transformed_data'

        constraint_mock.fit.assert_called_once_with('my_data')
        constraint_mock.transform.assert_called_once_with('my_data')

    def test_reverse_transform(self):
        """Test the ``Constraint.reverse_transform`` method. It is an identity method
        for completion, to be optionally overwritten by subclasses.

        The ``Constraint.reverse_transform`` method is expected to:
        - Return the input data unmodified.

        Input:
        - Anything
        Output:
        - Input
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        output = instance.reverse_transform('input')

        # Assert
        assert output == 'input'

    def test_is_valid(self):
        """Test the ``Constraint.is_valid` method. This should be overwritten by all the
        subclasses that have a way to decide which rows are valid and which are not.

        The ``Constraint.is_valid`` method is expected to:
        - Say whether the given table rows are valid.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_filter_valid(self):
        """Test the ``Constraint.filter_valid`` method.

        The ``Constraint.filter_valid`` method is expected to:
        - Filter the input data by calling the method ``is_valid``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data, with only the valid rows (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([True, True, False])

        # Run
        out = Constraint.filter_valid(constraint_mock, table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_filter_valid_with_invalid_index(self):
        """Test the ``Constraint.filter_valid`` method.

        Tests when the is_valid method returns a Series with an invalid index.

        Note: `is_valid.index` can be [0, 1, 5] if, for example, the Series is a subset
        of an original table with 10 rows, but only rows 0/1/5 were selected.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data, with only the valid rows (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        constraint_mock = Mock()
        is_valid = pd.Series([True, True, False])
        is_valid.index = [0, 1, 5]
        constraint_mock.is_valid.return_value = is_valid

        # Run
        out = Constraint.filter_valid(constraint_mock, table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_from_dict_fqn(self):
        """Test the ``Constraint.from_dict`` method passing a FQN.

        If the ``constraint`` string is a FQN, import the class
        before creating an instance of it.

        Input:
        - constraint dict with a FQN and args
        Output:
        - Instance of the subclass with the right args.
        """
        # Setup
        constraint_dict = {
            'constraint': 'sdv.constraints.tabular.FixedCombinations',
            'column_names': ['a', 'b'],
        }

        # Run
        instance = Constraint.from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, FixedCombinations)
        assert instance._columns == ['a', 'b']

    def test_from_dict_subclass(self):
        """Test the ``Constraint.from_dict`` method passing a subclass name.

        If the ``constraint`` string is a subclass name, take it from the
        Subclasses dict.

        Input:
        - constraint dict with a subclass name and args
        Output:
        - Instance of the subclass with the right args.
        """
        # Setup
        constraint_dict = {
            'constraint': 'FixedCombinations',
            'column_names': ['a', 'b'],
        }

        # Run
        instance = Constraint.from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, FixedCombinations)
        assert instance._columns == ['a', 'b']

    def test_to_dict(self):
        """Test the ``Constraint.to_dict`` method.

        The ``Constraint.to_dict`` method is expected to return a dict
        containting the FQN of the constraint instance and all the
        required arguments rebuild it.

        Output:
        - Dict with the right values.
        """
        # Run
        instance = FixedCombinations(column_names=['a', 'b'], handling_strategy='transform')
        constraint_dict = instance.to_dict()

        # Assert
        expected_dict = {
            'constraint': 'sdv.constraints.tabular.FixedCombinations',
            'handling_strategy': 'transform',
            'column_names': ['a', 'b'],
        }
        assert constraint_dict == expected_dict

    def test_to_dict_column_formula_lambda(self):
        """Test the ``Constraint.to_dict`` when the constraint is
        a ColumnFormula type and is passed a lambda.

        If the ``Constraint`` type is ColumnFormula,
        and the formula argument is a lambda, the dictionary
        should contain the lambda object as the value.

        Output:
        - Dict with the right values.
        """
        # Run
        instance = ColumnFormula(
            column='a',
            formula=lambda x: x + 1,
            handling_strategy='transform'
        )
        constraint_dict = instance.to_dict()

        # Assert
        assert constraint_dict['formula'](1) == 2

    def test_to_dict_column_formula_returned_function(self):
        """Test the ``Constraint.to_dict`` when the constraint is
        a ColumnFormula type and is passed a function returned
        from another function.

        If the ``Constraint`` type is ColumnFormula,
        and the formula argument is a function returned from another
        function, the dictionary should contain the function as the value.

        Output:
        - Dict with the right values.
        """
        # Run
        def func_creator():
            def func(x):
                return x + 1
            return func
        instance = ColumnFormula(
            column='a',
            formula=func_creator(),
            handling_strategy='transform'
        )
        constraint_dict = instance.to_dict()

        # Assert
        assert constraint_dict['formula'](1) == 2


class TestColumnsModel:

    def test___init__(self):
        """Test the ``__init__`` method of ``ColumnsModel``.

        Test that creating a ``ColumnsModel`` with a column name, stores it as a list.

        Setup:
            - Create an instance of ``ColumnsModel`` with a string representing the column
              name.

        Input:
            - String name of a column.

        Side Effects:
            - ``instance.constraint_columns`` is a list from the string given before.
        """

        # Run
        instance = ColumnsModel('age')

        # Assert
        assert instance.constraint_columns == ['age']

    def test___init__list(self):
        """Test the ``__init__`` method of ``ColumnsModel``.

        Test that creating a ``ColumnsModel`` with a list, stores it as a list

        Setup:
            - Create an instance of ``ColumnsModel`` with a list of strings.

        Input:
            - List of columns.

        Side Effects:
            - ``instance.constraint_columns`` is the input list.
        """

        # Run
        instance = ColumnsModel(['age', 'age_when_joined'])

        # Assert
        assert instance.constraint_columns == ['age', 'age_when_joined']

    @patch('sdv.constraints.base.GaussianMultivariate')
    @patch('sdv.constraints.base.HyperTransformer')
    def test_fit(self, mock_hyper_transformer, mock_gaussian_multivariate):
        """Test the ``fit`` method.

        The ``fit`` method should create an instance of ``rdt.HyperTransformer`` and use
        the ``fit_transform`` method in order to transform the data, which afterwards is being
        fitted to the ``GaussianMultivariate`` which uses as ``distribution`` the
        ``GaussianUnivariate``.

        Setup:
            - Instance of ``ColumnsModel``.

        Input:
            - table_data with 3 columns.

        Mock:
            - Mock ``rdt.HyperTransformer``.
            - Mock ``GaussianMultivariate``.

        Side Effects:
            - Instance of ``rdt.HyperTransformer`` has been created and fitted.
            - A ``GaussianMultivariate`` model has been created and fitted.
        """
        # Setup
        table_data = pd.DataFrame({
            'age': [1, 2, 3, 4],
            'age_when_joined': [5, 6, 7, 8],
            'retirement': ['a', 'b', 'c', 'd']
        })

        mock_hyper_transformer.return_value.fit_transform.return_value = 'transformed_data'

        instance = ColumnsModel(['age', 'age_when_joined'])

        # Run
        instance.fit(table_data)

        # Assert
        mock_hyper_transformer.assert_called_once_with(
            default_data_type_transformers={'categorical': 'OneHotEncodingTransformer'}
        )
        call_data = mock_hyper_transformer.return_value.fit_transform.call_args[0][0]
        pd.testing.assert_frame_equal(table_data[['age', 'age_when_joined']], call_data)

        mock_gaussian_multivariate.assert_called_once_with(distribution=GaussianUnivariate)
        mock_gaussian_multivariate.return_value.fit.assert_called_once_with('transformed_data')
        assert instance._hyper_transformer
        assert instance._model

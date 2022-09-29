"""Wrappers around copulas models."""
import warnings
from copy import deepcopy

import copulas
import copulas.multivariate
import copulas.univariate

from sdv.single_table.base import BaseSynthesizer


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a Truncated Gaussian distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    _DISTRIBUTIONS = {
        'norm': copulas.univariate.GaussianUnivariate,
        'beta': copulas.univariate.BetaUnivariate,
        'truncnorm': copulas.univariate.TruncatedGaussian,
        'gamma': copulas.univariate.GammaUnivariate,
        'uniform': copulas.univariate.UniformUnivariate,
        'gaussian_kde': copulas.univariate.GaussianKDE,
    }

    _model = None

    @classmethod
    def _validate_distribution(cls, distribution):
        if not isinstance(distribution, str) or distribution not in cls._DISTRIBUTIONS:
            error_message = f"Invalid distribution specification '{distribution}'."
            raise ValueError(error_message)

        return cls._DISTRIBUTIONS[distribution]

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 numerical_distributions=None, default_distribution=None):
        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        if numerical_distributions and not isinstance(numerical_distributions, dict):
            raise TypeError('numerical_distributions can only be None or a dict instance')

        self.default_distribution = default_distribution or 'beta'
        self.numerical_distributions = numerical_distributions or {}

        self._default_distribution = self._validate_distribution(self.default_distribution)
        self._numerical_distributions = {
            field: self._validate_distribution(distribution)
            for field, distribution in (numerical_distributions or {}).items()
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        numerical_distributions = deepcopy(self._numerical_distributions)

        for column in processed_data.columns:
            if column not in numerical_distributions:
                column_name = column.replace('.value', '')
                numerical_distributions[column] = self._numerical_distributions.get(
                    column_name, self._default_distribution)

        self._model = copulas.multivariate.GaussianMultivariate(
            distribution=numerical_distributions
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(processed_data)

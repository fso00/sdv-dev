"""Anonymization module for the ``DataProcessor``."""

import inspect

from faker import Faker
from rdt.transformers import AnonymizedFaker, get_default_transformers


SDTYPE_ANONYMIZERS = {
    'phisical_address': {
        'provider_name': 'address',
        'function_name': 'address'
    },
    'email': {
        'provider_name': 'internet',
        'function_name': 'email'
    },
    'latitude': {
        'provider_name': 'geo',
        'function_name': 'latitude'
    },
    'longitude': {
        'provider_name': 'geo',
        'function_name': 'longitude'
    },
    'ipv4_address': {
        'provider_name': 'internet',
        'function_name': 'ipv4'
    },
    'ipv6_address': {
        'provider_name': 'internet',
        'function_name': 'ipv6'
    },
    'mac_address': {
        'provider_name': 'internet',
        'function_name': 'mac_address'
    },
    'name': {
        'provider_name': 'person',
        'function_name': 'name'
    },
    'phone_number': {
        'provider_name': 'phone_number',
        'function_name': 'phone_number'
    },
    'ssn': {
        'provider_name': 'ssn',
        'function_name': 'ssn'
    },
    'user_agent_string': {
        'provider_name': 'user_agent',
        'function_name': 'user_agent'
    },
}


def _detect_provider_name(function_name):
    function_name = getattr(Faker(), function_name)
    module = inspect.getmodule(function_name).__name__
    module = module.split('.')
    if len(module) == 2:
        return 'BaseProvider'

    else:
        return module[-1]


def get_anonymized_transformer(function_name, function_kwargs=None):
    """Get an instance with an ``AnonymizedFaker`` for the given ``function_name``.

    Args:
        function_name (str):
            String representing predefined ``sdtype`` or a ``faker`` function.

        function_kwargs (dict):
            Keyword args to pass into the ``function_name``  when being called if needed.
            This is optional.
    """
    function_kwargs = function_kwargs or {}
    if function_name in SDTYPE_ANONYMIZERS:
        function_kwargs.update(SDTYPE_ANONYMIZERS[function_name])
        return AnonymizedFaker(**function_kwargs)

    provider_name = _detect_provider_name(function_name)
    function_kwargs.update({
        'function_name': function_name,
        'provider_name': provider_name
    })

    return AnonymizedFaker(**function_kwargs)

"""SDV versions."""

from importlib.metadata import version

public = version('sdv')
enterprise = None
connectors = None

__all__ = ('public', 'enterprise', 'connectors')

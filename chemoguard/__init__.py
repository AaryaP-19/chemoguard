"""Dataset and modelling layer for ChemoGuard.

The Flask application lives in the repository root as flat modules (``app.py``,
``cbc.py``, ``units.py``, ...). This package holds the parts that are about
*data* rather than about serving: ingestion, unit declaration at the source
boundary, and coherence filtering.

It imports the root modules (``units``, ``cbc``) rather than redefining
anything they own. ``Quantity.to`` stays the only conversion in the codebase.
"""

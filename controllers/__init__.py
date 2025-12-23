"""
Controller layer for Envista-Solomon.

Controllers coordinate between PyQt views under ``ui/`` and the service layer
under ``services/``. They do not own heavy business logic; instead they
orchestrate calls, persist UI state, and emit signals the views can render.
"""


"""The pipeline steps, in run order.

Each subpackage is one pipeline step; a step's canonical order and identity
live in its ``@register_stage(...)`` decorator (see ``stages/README.md`` for the
ordered map). Engine, plumbing, and user-facing surfaces stay at the top level
(``core/``, ``config/``, ``io/``, ``backends/``, ``methods/``, ``api/``,
``cli/``, ``visualization/``, ``utils/``); this package is only the analysis
steps.
"""

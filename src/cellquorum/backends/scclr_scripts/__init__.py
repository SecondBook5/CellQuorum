"""In-env helper scripts executed inside the isolated scclr environment.

These scripts are NOT imported by CellQuorum — they are invoked as subprocesses
via ``ScclrBackend.run_helper`` and run under the scclr env's interpreter, where
``import scclr`` is available.
"""

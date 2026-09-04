"""In-env helper scripts executed inside the isolated partipy environment.

These scripts are NOT imported by CellQuorum — they are invoked as subprocesses via
``PartipyBackend.run_helper`` and run under the partipy env's interpreter, where
``import partipy`` is available.

The isolation is a licensing boundary as well as a dependency one: partipy is GPL-3 and
CellQuorum is BSD-3, so importing it in-process would make the combined distribution copyleft.
Two programs exchanging files do not have that problem.

This package exists so the helper actually ships in a wheel: ``package-data`` only covers
``r_scripts/*.R``, and every other in-env script directory is discovered as a package instead.
Without this file the helper is present in a git checkout and silently absent from an install.
"""

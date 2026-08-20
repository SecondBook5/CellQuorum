"""QC diagnostic and publication figure builders (moved from the ``qc/`` package, Move 2).

Import the specific module you need — ``cellquorum.visualization.qc.diagnostics`` for the
standard audit plots, or ``cellquorum.visualization.qc.publication`` for the journal-style
panels. Importing those modules (not this package) is what pulls in matplotlib/seaborn,
so a caller loads the plotting stack only when it actually renders a figure.
"""

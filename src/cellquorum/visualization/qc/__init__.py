"""QC figure and table builders.

Import the specific module you need — ``graded`` for the figures that describe the graded
adjudication (evidence families, states, eligibility), ``panels`` for the cohort overview set, or
``publication_table`` for the typeset sample-level tables. Importing those modules (not this
package) is what pulls in matplotlib/seaborn, so a caller loads the plotting stack only when it
actually renders a figure.

``diagnostics`` and ``publication`` were removed with the v1 threshold path. Both keyed on the
``cellquorum_qc_keep`` verdict, which graded adjudication does not produce, so neither could
render a graded run at all; ``graded`` replaces them.
"""

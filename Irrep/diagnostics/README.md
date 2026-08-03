# Diagnostics

This directory holds exploratory or historical symmetry checks.  The active
IrRep workflow remains in `Irrep/scripts/`.

`cno_site_symmetry_legacy.py` is retained only as the original standalone
site-symmetry exploration.  It assumes the former one-to-one WS sample map
and must not be used for finite-volume regional CNOs.  The supported CNO-field
checker is `Density matrix cal/symmetry/cno_symmetry.py`; it understands the
saved weighted quadrature contract and tests CNO subspace closure.

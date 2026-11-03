# Reference data

`generated/` contains the supplied numerical tables, profiles, exact results,
metadata and four paper figures. These are reference outputs, retained from
the recorded calculations. The directory name reflects how they were made;
normal runs now write to `results/data/` instead.

See [the output map](../docs/PAPER_OUTPUTS.md) for the command and input data
behind each figure and table, and [validation coverage](../docs/VALIDATION.md)
for the checks actually performed. Individual metadata files record input
checksums and the numerical scope where applicable.

Numerical values and reference figure files are unchanged by this repository
cleanup. Private checkout paths in metadata have been replaced with
repository-relative paths. Legacy data field names such as `gap_over_Sstar`
are retained for compatibility; `Sstar` in these fields denotes the paper's
extremal reference entropy, `S_ext`.

# Validation records

`reference/` contains the historical run logs, environments and summaries
supplied with the reference data. These records describe the tasks executed
at the time of each run; they are not a claim that every task in the current
runner was executed together. See [validation coverage](../docs/VALIDATION.md).

The repository cleanup changes path strings only in retained records:

- Package paths are repository-relative.
- `[validation-host]` replaces private machine directories.
- `[external-manuscript]` replaces the destination of an earlier figure export.
- Record directories have descriptive names instead of editorial revision names.

Reported scientific values, tolerances, timings and exit codes are retained.
Warnings in historical logs remain part of those records. Some task lists
describe older runner versions and are deliberately preserved as historical
coverage, rather than updated to imply additional tests.

New runs write to the ignored `results/validation/` directory. The repository
manifest excludes `results/`, so rerunning a calculation does not invalidate
the reference snapshot.

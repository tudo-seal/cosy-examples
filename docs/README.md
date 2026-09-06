# Known limits

Three files, and each one collects the things a reader would otherwise have to find out by running
into them. None of them is a bug list. What stands here is a property of the method or of a search
space rather than a defect waiting to be fixed, and knowing it in advance is the difference, and
knowing it in advance is the difference
between reading a result and misreading one.

| File | What it covers |
|---|---|
| [`surrogate-limits.md`](surrogate-limits.md) | the loop, the model and the five reads |
| [`search-space-limits.md`](search-space-limits.md) | what a space must admit to be searched |
| [`cnn-example-limits.md`](cnn-example-limits.md) | the CNN architecture search in particular |

Every statement in the three files is a statement about code in this repository, and the file names
the module it can be read in. Where a statement is a measurement, it names the configuration it was
measured under in the same sentence, because none of these numbers is a property of the method
alone.

Defects of the one recorded run are not here. They belong to that run, they are listed in
`README.md` and asserted in `tests/test_recorded_run.py`, and they are pinned rather than repaired
because the files in `recorded_runs/` are evidence.

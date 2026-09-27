"""Where the artifacts of a run go: beside its CSV, named after it."""

import os


def metadata_path_for(csv_path):
    """Return the provenance path beside a run's CSV: ``run.csv`` becomes ``run_config.json``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}_config.json"


def _sibling_path(csv_path, suffix):
    """Return a path beside a run's CSV: ``run.csv`` becomes ``run<suffix>``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}{suffix}"


class RunArtifacts:
    """Every file a run writes, named after its CSV.

    A run writes seven files beside each other, and a second start against the same name would
    truncate the ones a first run is still writing, since every writer opens in ``"w"`` mode.  Named
    here in one place, they can be checked all at once before anything is opened.

    Args:
        stem (str): The run's CSV; every other file is named after it.
    """

    #: The files of a run by kind, as suffixes of the CSV's name without its extension.
    SUFFIXES = {
        "terms": "_terms.pickle",
        "ea": "_ea.csv",
        "surrogate": "_surrogate.csv",
        "config": "_config.json",
        "diagnostics": "_diagnostics.json",
        "trace": "_trace.csv",
    }

    def __init__(self, stem):
        self.stem = str(stem)

    def path(self, kind):
        """The path of one kind of file: ``"csv"`` or one of :attr:`SUFFIXES`."""
        return self.stem if kind == "csv" else _sibling_path(self.stem, self.SUFFIXES[kind])

    def paths(self):
        """Every file of the run by kind, the CSV first."""
        return {"csv": self.stem, **{kind: self.path(kind) for kind in self.SUFFIXES}}

    def taken(self):
        """The files of the run that already exist."""
        return [path for path in self.paths().values() if os.path.exists(path)]

    def refuse_taken(self):
        """Refuse a run whose files exist, naming every one, before anything is opened.

        Raises:
            FileExistsError: If any file of the run exists.
        """
        taken = self.taken()
        if taken:
            msg = (
                f"another run's files exist under this name: {', '.join(taken)}; a second start "
                "is refused rather than truncating a run that may still be writing"
            )
            raise FileExistsError(msg)

#!/bin/bash
# Run the search that produced the records beside this script, or one of the two tutorial cells
# under the same settings.
#
# The recorded run is bo_VGGM_20260805_r3. Every setting passed below except --protocol and
# --data-dir is also written to bo_VGGM_20260805_r3_config.json, so a new run and the recorded one
# can be compared field by field instead of by memory. The training recipe is not in that file, so
# a rerun that changes --protocol leaves no trace in the record.
#
# Usage, from the repository root:
#
#     recorded_runs/bo_20260805_r3.sh <VGGM|TUT1|TUT2> [<tag>] [further experiment arguments]
#
# The tag names the artifacts. A run that changes a setting needs a tag of its own, or it
# overwrites the run it is meant to be compared with.
#
# One thing this script does not do that the recorded run did: it trains its own initial design.
# The record's 20 initial terms were taken over from an earlier attempt with --resume-from, which
# is why their timestamps lie within a second of each other while their trainings sum to more than
# four hours. Append --resume-from <terms.pickle> after the tag to do the same.
#
# Arguments after the tag are appended to the command line and override what this script sets,
# which is how a smoke test shortens the budget:
#
#     recorded_runs/bo_20260805_r3.sh VGGM smoke --epochs 2 --n-pre-samples 2 --n-iterations 1 \
#         --population-size 5 --evo-generations 3 --repeats 1
#
# What one run costs. The recorded VGGM run spent 75287 s in the optimization loop on one NVIDIA
# A30. Of that, 58670 s went into its 30 acquisition maximizations and 16556 s into training the
# 30 candidates they picked, so 78 percent of the loop is the acquisition and a longer run costs
# more of that rather than more training. The other trainings of the record lie outside those
# 75287 s: 15779 s of them were paid by an earlier run whose initial design this one took over,
# and 10736 s by the random arm, which runs after the loop. Those are the numbers of one cell on
# one card.
set -u

CELL="${1:?the cell is missing: VGGM, TUT1 or TUT2}"
TAG="${2:-bo_${CELL}_20260805_r3}"
shift $(( $# > 2 ? 2 : $# ))

# Where torchvision keeps the CIFAR-10 archive. The experiment does not fetch it. A missing
# archive stops the run at its start rather than pulling 170 MB in the middle
# of a search that has been going for hours.
DATA_DIR="${CIFAR10_DIR:-./data}"

# Seconds after which one acquisition maximization is given up on. The default of the experiment
# is 3600. This run raised it to 14400 and never came near it: the longest of its 30 acquisition
# maximizations took 3283 s. The limit bounds a legitimate duration, so it is a property of the
# machine and of the cell together, and a slower card needs it raised again.
HARD_LIMIT=14400

# The bayesian_optimization package is imported from the working directory, so this script runs
# from the repository root and nowhere else. The root is put in front of whatever the caller has
# already set rather than replacing it, so an installation that keeps cosy on PYTHONPATH keeps it.
if [ ! -d "bayesian_optimization" ]; then
  echo "run this from the repository root: bayesian_optimization/ is not here" >&2
  exit 1
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Lets the CUDA allocator give memory back between candidates. The terms of these cells differ by
# an order of magnitude in width, and without it the allocator fragments across a long run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results

# Two processes writing one artifact overwrite each other's lines instead of failing, and a search
# that runs for a day is started by hand often enough for that to happen. The lock is a directory
# because creating one is atomic.
LOCK="results/.${TAG}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$LOCK exists: is this run already going?" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "=== $TAG started: $(date) ==="

# --sampling depth-bounded is not a preference. The command line defaults to size-uniform, which
# determinizes the search space and draws from the determinized program, and that is a different
# sampler than the recorded run used. A run that leaves the option out is therefore not a variant
# of this one, it is a different search, and on a space whose determinized stream runs dry it ends
# in the RuntimeError that names how many distinct inhabitants it collected.
#
# --repeats 3 makes every objective value the mean of three trainings, under the seeds 0 to 2.
# Training noise can be wider than the difference between two candidates, in which case one
# training cannot decide between them and the mean over three divides that noise by the square
# root of three.
#
# --baseline runs a random search of the same budget from the same initial design, into the same
# artifacts. It is what makes the comparison paired, and it costs one extra evaluation per
# iteration rather than a second full run, which under --repeats 3 is three more trainings.
STATUS=0
"${PYTHON:-python3}" -u -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment \
  --target "$CELL" \
  --data-dir "$DATA_DIR" \
  --acquisition-hard-limit "$HARD_LIMIT" \
  --n-pre-samples 20 --n-iterations 30 \
  --population-size 100 --evo-generations 35 \
  --crossover-rate 0.9 --mutation-rate 0.15 \
  --repeats 3 \
  --epochs 50 --batch-size 128 \
  --sampling depth-bounded \
  --kernel damg --protocol corrected --seed 0 \
  --acquisition ExpectedImprovement \
  --baseline \
  --csv-path "results/${TAG}.csv" \
  "$@" || STATUS=$?

# The exit status is passed on. A runner that reports success whichever way the run ended is how a
# failed start gets mistaken for a finished search.
echo "=== $TAG finished: $(date), exit status $STATUS ==="
exit "$STATUS"

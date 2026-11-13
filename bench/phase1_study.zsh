#!/bin/zsh
# The decisive test of the neural-harness study's Phase 1 (its RESULTS-P0.md, 6.1), as three runs of bench.phase1.
#
# What it does
#   Runs three arms one after another, each in its own process and its own ledger. A row may write 32,768 tokens:
#     nothink   the base model with thinking off
#     edited    the edited model, thinking on, and a forced exit at 4,096, 8,192, 16,384 and 24,576 tokens
#     base      the base model, thinking on, with the same four forced exits; its rows stop at 24,577 tokens,
#               one past the last exit point (the rest of a base row serves no condition of the test)
#   A forced exit may write 2,000 tokens after the exit phrase. Every arm is graded with math-verify and keeps
#   the batch at one width, so a row can be repeated from its seed. If an arm fails, nothing after it starts.
#   The short arm goes first, the base arm last: its rows nearly all run to their cap and finish in the order
#   they started, so a run that is cut short still leaves whole problems from the front of the order.
#   Last, bench.phase1_analysis makes the comparisons between the arms that are listed at the end of this file.
#   The Hugging Face libraries are told to stay offline: the problems and the model are already on this disk,
#   and a run of many hours must not depend on the network or pick up a changed dataset half-way.
#
# Why a script
#   The plan written before the run names these commands. Running them from one file keeps the run and the
#   plan the same thing, and lets the run be repeated.
#
# Run:     uv run python -m lab.run --name phase1 --max-hours <limit> -- \
#              zsh bench/phase1_study.zsh <name> <plan file> <problems> <samples per problem> <run seed>
#          (lab.run keeps the Mac awake, logs to runs/logs/ and stops the run if memory runs out or the limit passes)
#          Started again with the same arguments, it continues: finished rows are in the ledgers.
# Changes: runs/phase1/<name>-nothink, <name>-edited and <name>-base (ledgers), three entries of those names
#          in bench/results/phase1.json and one in bench/results/phase1_analysis.json. Reads the study's folder,
#          never writes to it.

cd "${0:A:h}/.." || exit 1
if (( $# != 5 )); then
  echo "use: zsh bench/phase1_study.zsh <name> <plan file> <problems> <samples per problem> <run seed>"
  exit 2
fi
name=$1 plan=$2 problems=$3 samples=$4 seed=$5
exits=4096,8192,16384,24576
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# One arm: one process. Its other arguments are the ones that differ between arms.
arm() {
  local tag=$name-$1
  shift
  echo "== $tag: started $(date '+%Y-%m-%d %H:%M:%S')"
  uv run python -m bench.phase1 --tag $tag --plan $plan --n $problems --samples $samples --run-seed $seed \
      --grader math-verify --keep-width "$@"
  local code=$?
  if (( code != 0 )); then
    echo "== $tag: failed with exit code $code; nothing after it is started"
    exit 1
  fi
  echo "== $tag: finished $(date '+%Y-%m-%d %H:%M:%S')"
}

arm nothink --arm base --no-thinking
arm edited --arm edited --exits $exits --answer-tokens 2000
arm base --arm base --trunk-tokens 24577 --exits $exits --answer-tokens 2000 --keep-free 16
echo "== all three arms finished $(date '+%Y-%m-%d %H:%M:%S')"

# The comparisons. The first is the primary one: the edited model at its full cap against the base model's
# forced exit of about the same token cost. Then the edited model against thinking off (at the same cap, and
# cut at Phase 0's cap of 8,192) and against the next cheaper forced exit, and the two models under the same
# forced exit at each of the four points.
uv run python -m bench.phase1_analysis --name $name \
    --compare edited@32768,base+exit@24576 \
    --compare edited@32768,nothink@32768 \
    --compare edited@32768,nothink@8192 \
    --compare edited@32768,base+exit@16384 \
    --compare edited+exit@4096,base+exit@4096 \
    --compare edited+exit@8192,base+exit@8192 \
    --compare edited+exit@16384,base+exit@16384 \
    --compare edited+exit@24576,base+exit@24576

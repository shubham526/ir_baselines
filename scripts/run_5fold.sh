#!/usr/bin/env bash
# Five-fold train / test / concatenate for one model.
#
#   MODEL=rankt5 DATA=/path/to/fold_data QRELS=/path/to/qrels.txt \
#   OUT=/path/to/runs bash scripts/run_5fold.sh
#
# DATA must contain fold-0 .. fold-4, each with training.jsonl,
# validation.jsonl and testing.jsonl. VAL_QRELS points at per-fold validation
# qrels; if unset, the full qrels is used for validation too, which is wrong
# for checkpoint selection and will be refused by require_full_coverage.
#
# Stops at the first failure. The concatenated run is checked against the
# qrels topic set before it is reported: a short concatenation still evaluates
# cleanly and returns a plausible number.
set -euo pipefail

MODEL="${MODEL:?set MODEL -- see: python -m ir_baselines.train --list-models}"
DATA="${DATA:?set DATA to the fold data root}"
QRELS="${QRELS:?set QRELS to the full qrels}"
VAL_QRELS="${VAL_QRELS:-}"
OUT="${OUT:-./runs}"
CKPT="${CKPT:-./checkpoints}"

PRETRAIN="${PRETRAIN:-}"          # empty = the model's default encoder
LOSS="${LOSS:-}"                  # empty = the model's default objective
EXTRA="${EXTRA:-}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-5}"
MAXLEN="${MAXLEN:-512}"           # pair encoding
QLEN="${QLEN:-20}"                # dual encoding
DLEN="${DLEN:-512}"               # dual encoding
SEED="${SEED:-42}"
CUDA="${CUDA:-0}"
AMP="${AMP:-bf16}"
TREC_EVAL="${TREC_EVAL:-trec_eval}"
TREC_FLAGS="${TREC_FLAGS:--c}"

TAG="$MODEL"
mkdir -p "$OUT/$TAG" "$CKPT/$TAG"

# Optional flags are assembled rather than interpolated, so an empty value
# does not become an empty positional argument.
OPT=()
[ -n "$PRETRAIN" ] && OPT+=(--pretrain "$PRETRAIN")
[ -n "$LOSS" ] && OPT+=(--loss "$LOSS")
LEN_OPT=(--max-len "$MAXLEN" --max-query-len "$QLEN" --max-doc-len "$DLEN")

for FOLD in 0 1 2 3 4; do
  FOLD_QRELS="${VAL_QRELS:+$VAL_QRELS/fold-$FOLD/validation.qrels.txt}"
  FOLD_QRELS="${FOLD_QRELS:-$QRELS}"

  echo "=================== $TAG fold-$FOLD : train"
  python -m ir_baselines.train \
    --model "$MODEL" "${OPT[@]}" $EXTRA \
    --train "$DATA/fold-$FOLD/training.jsonl" \
    --dev   "$DATA/fold-$FOLD/validation.jsonl" \
    --qrels "$FOLD_QRELS" \
    --save-dir "$CKPT/$TAG/fold-$FOLD" \
    "${LEN_OPT[@]}" \
    --batch-size "$BATCH" --epoch "$EPOCHS" --learning-rate "$LR" \
    --seed "$SEED" --amp "$AMP" --use-cuda --cuda "$CUDA"

  echo "=================== $TAG fold-$FOLD : test"
  python -m ir_baselines.test \
    --model "$MODEL" "${OPT[@]}" $EXTRA \
    --test "$DATA/fold-$FOLD/testing.jsonl" \
    --checkpoint "$CKPT/$TAG/fold-$FOLD/model.bin" \
    --save-dir "$OUT/$TAG" --run "fold-$FOLD.run" \
    --run-tag "$MODEL" \
    "${LEN_OPT[@]}" \
    --batch-size "$BATCH" --use-cuda --cuda "$CUDA"
done

# Concatenate. This is the step where a missing fold goes unnoticed: the
# master run is short, and it still scores.
MASTER="$OUT/$TAG/master.run"
cat "$OUT/$TAG"/fold-*.run > "$MASTER"

EXPECTED=$(cut -d' ' -f1 "$QRELS" | sort -u | wc -l)
GOT=$(awk '{print $1}' "$MASTER" | sort -u | wc -l)
echo
echo "master run : $MASTER"
echo "topics     : $GOT (qrels has $EXPECTED)"
if [ "$GOT" -ne "$EXPECTED" ]; then
  echo "ERROR  topic count mismatch. Missing topics:"
  comm -23 <(cut -d' ' -f1 "$QRELS" | sort -u) <(awk '{print $1}' "$MASTER" | sort -u) | head
  exit 1
fi

echo
echo "=================== $TAG scored with $TREC_EVAL $TREC_FLAGS"
"$TREC_EVAL" $TREC_FLAGS -m map -m ndcg_cut.20 -m P.20 -m recip_rank "$QRELS" "$MASTER"
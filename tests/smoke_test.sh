#!/usr/bin/env bash
# Exit-code assertions for the whole pipeline.
#
# Every check asserts the exit code rather than grepping stdout. A narrow grep
# will happily report success on a run that wrote its output file and then
# crashed during scoring, which is how an unpack mismatch in test.py survived
# an earlier round of testing.
#
#   PRETRAIN=bert DATA=/path/to/tiny/fold bash tests/smoke_test.sh
#
# DATA must contain train.jsonl, dev.jsonl, test.jsonl, qrels.txt. A few dozen
# examples over a handful of topics is enough; this checks wiring, not quality.
set -u

PRETRAIN="${PRETRAIN:-bert}"
DATA="${DATA:-./smoke_data}"
OUT="${OUT:-/tmp/irb_smoke}"
MAXLEN="${MAXLEN:-64}"
QLEN="${QLEN:-16}"
DLEN="${DLEN:-48}"
NTOPICS="${NTOPICS:-}"

FAIL=0
run() {   # run <ok|fail> <label> <cmd...>
  local expect=$1 label=$2; shift 2
  local out code
  out=$("$@" 2>&1); code=$?
  if [ "$expect" = ok ] && [ $code -ne 0 ]; then
    echo "FAIL  $label (exit $code)"; echo "$out" | tail -5 | sed 's/^/        /'; FAIL=1
  elif [ "$expect" = fail ] && [ $code -eq 0 ]; then
    echo "FAIL  $label (expected non-zero exit, got 0)"; FAIL=1
  else
    echo "PASS  $label"
    if [ "$expect" = ok ]; then
      echo "$out" | grep -E "Scored|  map |Training complete|pairs written" | sed 's/^/        /'
    fi
  fi
}

rm -rf "$OUT"; mkdir -p "$OUT"
TOPIC_FLAG=(); [ -n "$NTOPICS" ] && TOPIC_FLAG=(--expected-topics "$NTOPICS")

echo "--- cross-encoder"
run ok "train bert" python -m ir_baselines.train --model bert --pretrain "$PRETRAIN" \
  --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" \
  --save-dir "$OUT/bert" --max-len "$MAXLEN" --batch-size 4 --epoch 2 --seed 42
run ok "test bert with --qrels" python -m ir_baselines.test --model bert --pretrain "$PRETRAIN" \
  --test "$DATA/test.jsonl" --save-dir "$OUT/runs" --run bert.run \
  --checkpoint "$OUT/bert/model.bin" --max-len "$MAXLEN" \
  --qrels "$DATA/qrels.txt" "${TOPIC_FLAG[@]}"

echo "--- multi-vector"
for M in me-bert poly-encoder; do
  run ok "train $M" python -m ir_baselines.train --model "$M" --pretrain "$PRETRAIN" \
    --logit-scale --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" \
    --qrels "$DATA/qrels.txt" --save-dir "$OUT/$M" \
    --max-query-len "$QLEN" --max-doc-len "$DLEN" --batch-size 4 --epoch 2 --seed 42
  run ok "test $M with --qrels" python -m ir_baselines.test --model "$M" --pretrain "$PRETRAIN" \
    --logit-scale --test "$DATA/test.jsonl" --save-dir "$OUT/runs" --run "$M.run" \
    --checkpoint "$OUT/$M/model.bin" --max-query-len "$QLEN" --max-doc-len "$DLEN" \
    --qrels "$DATA/qrels.txt" "${TOPIC_FLAG[@]}"
done

run ok "ce-inbatch objective" python -m ir_baselines.train --model me-bert --pretrain "$PRETRAIN" \
  --loss ce-inbatch --positives-only --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" \
  --qrels "$DATA/qrels.txt" --save-dir "$OUT/ce" \
  --max-query-len "$QLEN" --max-doc-len "$DLEN" --batch-size 5 --epoch 1

run ok "init-checkpoint carrying config" python -m ir_baselines.train --model me-bert \
  --pretrain "$PRETRAIN" --logit-scale --init-checkpoint "$OUT/me-bert/model.bin" \
  --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" \
  --save-dir "$OUT/init" --max-query-len "$QLEN" --max-doc-len "$DLEN" \
  --batch-size 4 --epoch 1

echo "--- determinism"
for i in 1 2; do
  python -m ir_baselines.train --model bert --pretrain "$PRETRAIN" \
    --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" \
    --save-dir "$OUT/det$i" --max-len "$MAXLEN" --batch-size 4 --epoch 2 --seed 42 \
    > /dev/null 2>&1
  python -m ir_baselines.test --model bert --pretrain "$PRETRAIN" \
    --test "$DATA/test.jsonl" --save-dir "$OUT/det$i" --run out.run \
    --checkpoint "$OUT/det$i/model.bin" --max-len "$MAXLEN" > /dev/null 2>&1
done
if cmp -s "$OUT/det1/out.run" "$OUT/det2/out.run"; then
  echo "PASS  same seed gives byte-identical runs"
else
  echo "FAIL  same seed gave different runs"; FAIL=1
fi

echo "--- run provenance"
if [ -f "$OUT/runs/bert.run.provenance.json" ]; then
  echo "PASS  sibling provenance written beside the run"
else
  echo "FAIL  no sibling provenance beside the run"; FAIL=1
fi
if python -c "
import json, sys
from ir_baselines import provenance as P
rec = json.load(open('$OUT/runs/bert.run.provenance.json'))
sys.exit(0 if rec['run']['sha256'] == P.run_stats('$OUT/runs/bert.run')['sha256'] else 1)"; then
  echo "PASS  recorded digest matches the run file"
else
  echo "FAIL  recorded digest does not match the run file"; FAIL=1
fi
run ok "inspect a run" python -m ir_baselines.inspect "$OUT/runs/bert.run"
run ok "inspect a checkpoint" python -m ir_baselines.inspect "$OUT/bert/model.bin"

echo "--- resume"
run ok "resume from last.bin" python -m ir_baselines.train --model bert \
  --pretrain "$PRETRAIN" --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" \
  --qrels "$DATA/qrels.txt" --save-dir "$OUT/bert" --max-len "$MAXLEN" \
  --batch-size 4 --epoch 4 --seed 42 --resume "$OUT/bert/last.bin"
if python -c "
import json, sys
h = json.load(open('$OUT/bert/training_history.json'))
sys.exit(0 if h['epoch'] == [1, 2, 3, 4] else 1)"; then
  echo "PASS  resumed history continues rather than restarting"
else
  echo "FAIL  resumed history did not continue"; FAIL=1
fi

python -c "
import torch
o = torch.load('$OUT/me-bert/model.bin', weights_only=False)
torch.save(o['model_state_dict'], '$OUT/bare.bin')
torch.save({'model_state_dict': o['model_state_dict'], 'config': o['config']},
           '$OUT/noopt.bin')" \
  || { echo "FAIL  could not build legacy checkpoints"; FAIL=1; }

echo "--- guards (each must exit non-zero)"
run fail "init-checkpoint without config refused" python -m ir_baselines.train --model me-bert \
  --pretrain "$PRETRAIN" --logit-scale --init-checkpoint "$OUT/bare.bin" \
  --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" \
  --save-dir "$OUT/x" --max-query-len "$QLEN" --max-doc-len "$DLEN" --epoch 1
run fail "bce + positives-only refused" python -m ir_baselines.train --model me-bert \
  --pretrain "$PRETRAIN" --loss bce --positives-only --train "$DATA/train.jsonl" \
  --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" --save-dir "$OUT/x"
run fail "inference without checkpoint refused" python -m ir_baselines.test --model me-bert \
  --pretrain "$PRETRAIN" --test "$DATA/test.jsonl" --save-dir "$OUT/x" --run x.run
run fail "unverified checkpoint refused" python -m ir_baselines.test --model me-bert \
  --pretrain "$PRETRAIN" --logit-scale --test "$DATA/test.jsonl" --save-dir "$OUT/x" \
  --run x.run --checkpoint "$OUT/bare.bin" --max-query-len "$QLEN" --max-doc-len "$DLEN"
run fail "tower-topology mismatch refused" python -m ir_baselines.test --model me-bert \
  --pretrain "$PRETRAIN" --logit-scale --shared-encoder false --test "$DATA/test.jsonl" \
  --save-dir "$OUT/x" --run x.run --checkpoint "$OUT/me-bert/model.bin" \
  --max-query-len "$QLEN" --max-doc-len "$DLEN"
run fail "length mismatch refused" python -m ir_baselines.test --model bert \
  --pretrain "$PRETRAIN" --test "$DATA/test.jsonl" --save-dir "$OUT/x" --run x.run \
  --checkpoint "$OUT/bert/model.bin" --max-len 32
run fail "objective the model cannot consume refused" python -m ir_baselines.train \
  --model bert --pretrain "$PRETRAIN" --loss bce --train "$DATA/train.jsonl" \
  --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" --save-dir "$OUT/x"
run fail "resume without training state refused" python -m ir_baselines.train \
  --model me-bert --pretrain "$PRETRAIN" --logit-scale --resume "$OUT/noopt.bin" \
  --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" \
  --save-dir "$OUT/x" --max-query-len "$QLEN" --max-doc-len "$DLEN" --epoch 9
run fail "--resume with --init-checkpoint refused" python -m ir_baselines.train \
  --model bert --pretrain "$PRETRAIN" --resume "$OUT/bert/last.bin" \
  --init-checkpoint "$OUT/bert/last.bin" --train "$DATA/train.jsonl" \
  --dev "$DATA/dev.jsonl" --qrels "$DATA/qrels.txt" --save-dir "$OUT/x"

echo
if [ $FAIL -eq 0 ]; then echo "SMOKE SUITE PASSED"; else echo "SMOKE SUITE FAILED"; fi
exit $FAIL

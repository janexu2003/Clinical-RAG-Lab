#!/usr/bin/env bash
# Measures what the cross-encoder and the section boost each add (retrieval only, no API calls).
#   bash scripts/run_rerank_ablation.sh
set -e
cd "$(dirname "$0")/.."
for c in r0_no_rerank r1_section_boost r2_cross_encoder r3_cross_encoder_boost; do
  echo "=== $c"
  python scripts/run_eval.py --config configs/rerank_ablation/$c.yaml --retrieval-only > results_$c.log 2>&1 \
    && grep -E "^\| (hit@1|hit@3|hit@5|mrr|ndcg@5|latency_s) " results/${c}_retrieval/report.md \
    || { echo "FAILED - see results_$c.log"; exit 1; }
done
M="hit@1 hit@3 hit@5 mrr ndcg@5"
echo; echo "##### effect of the cross-encoder alone (r0 -> r2)"
python scripts/compare_runs.py results/r0_no_rerank_retrieval results/r2_cross_encoder_retrieval --metrics $M
echo; echo "##### effect of the section boost alone (r0 -> r1)"
python scripts/compare_runs.py results/r0_no_rerank_retrieval results/r1_section_boost_retrieval --metrics $M
echo; echo "##### adding the boost on top of the cross-encoder (r2 -> r3)"
python scripts/compare_runs.py results/r2_cross_encoder_retrieval results/r3_cross_encoder_boost_retrieval --metrics $M
rm -f results_r*.log

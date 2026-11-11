# clinical-rag-lab: summary of results

**Goal:** a medical RAG chatbot over one textbook (Gale Encyclopedia of Medicine), with three questions:
1. How do you diagnose and fix poor retrieval?
2. How do you prevent and detect hallucination?
3. How do you evaluate it, on your own data and on established benchmarks?

All numbers below are from runs on a MacBook Air with gpt-4o-mini. 95% CIs come from bootstrap; differences marked * are significant (paired test, CI excludes 0).

## What was built
| Area | Method |
|---|---|
| Chunking | Footer/caption removal; entry + section chunks with a contextual header ("Acne \| Diagnosis") |
| Retrieval | BM25 + MiniLM dense (hybrid, RRF fusion); cross-encoder reranking; section boost; neighbour expansion |
| Diagnosis | Each failure classified as NOT_IN_CORPUS / RECALL_MISS / CANDIDATE_MISS / RANKING_MISS / generation problem |
| Hallucination | Grounded prompt with `[n]` citations; free checks (numbers, citations, word overlap, exact quotes); LLM claim verification (faithfulness); repair |
| Evaluation | Retrieval metrics (Hit@k, MRR, nDCG), LLM-judged correctness, abstention rates, bootstrap CIs, paired comparisons |

## Results

### A. Own textbook: 30 in-scope + 5 out-of-scope questions
| | v1 (original chatbot) | improved | Difference |
|---|---|---|---|
| Hit@1 | 0.40 | 0.87 | +0.47* |
| MRR | 0.58 | 0.93 | +0.35* |
| Faithfulness | 0.88 | 1.00 | +0.12* |
| Correctness | 0.93 | 0.92 | n.s. |
| Out-of-scope declined | 5/5 | 5/5 | |
| False abstentions | 0 | 2 | |

Reranker ablation (same hybrid setup, only the reranking step changes):
- **Cross-encoder:** MRR 0.82 → 0.93 (+0.12*), at about 0.1 s per question.
- **Section boost:** no measurable effect once dense search and the cross-encoder are used.

### B. BEIR NFCorpus: retrieval only, 323 queries, 3,633 abstracts
| | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| Published BM25 (BEIR paper) | 0.325 | – | – |
| BM25 (ours) | 0.318 | 0.240 | 0.520 |
| Hybrid | 0.352* | 0.323* | 0.558* |
| Hybrid + cross-encoder | 0.353 (n.s.) | 0.323 | 0.581 (n.s.) |

### C. MIRAGE: answer accuracy, 500 questions (100 × 5 datasets), MedRAG Textbooks corpus
| | All | BioASQ | MedMCQA | MedQA | MMLU-Med | PubMedQA* |
|---|---|---|---|---|---|---|
| No retrieval (CoT) | **0.774** | 0.83 | 0.78 | 0.78 | 0.90 | 0.58 |
| RAG, BM25 over textbooks | 0.732 (−0.042*) | 0.72 | 0.73 | 0.80 | 0.89 | 0.52 |

## Key findings
1. **Retrieval quality was the main problem in v1, and it's fixable.** Structure-aware chunks with contextual headers, hybrid search and a cross-encoder took Hit@1 from 0.40 to 0.87 on the textbook.
2. **Grounding prevents hallucination, but can over-abstain.** Faithfulness reached 1.00 and all out-of-scope questions were declined. However, the strict prompt refused two false-premise questions it could have answered ("Which foods cause acne?").
3. **Free checks catch form, not meaning.** Invented numbers, invalid citations and fabricated quotes are caught for free; a claim that reuses the source's words with the wrong meaning ("cures acne permanently") needs the LLM check. "Unsupported" ≠ "false": it can also mean the right chunk wasn't retrieved.
4. **Gains depend on the domain.** Hybrid search helped on both the textbook and NFCorpus. The cross-encoder (trained on web search) helped on the textbook but not on NFCorpus, whose labels follow NutritionFacts.org citation links rather than topical relevance.
5. **The wrong corpus makes RAG worse than no RAG.** Textbook retrieval lowered gpt-4o-mini's accuracy by 4.2 points, mainly on research questions (BioASQ, PubMedQA*) that textbooks don't cover, and used 12× more input tokens. This matches the MedRAG paper (Table 7). The knowledge source must contain the answers.
6. **Small eval sets mislead.** On 30 textbook questions, most offline gains were not significant. Benchmarks with 300–500 questions and paired tests give trustworthy conclusions.

## Limitations
- One encyclopedia volume (A–B, 2004); 30 in-scope questions, 20 of them written from the book text.
- gpt-4o-mini is both generator and judge; judge agreement with human labels has not been checked yet (`scripts/judge_agreement.py`).
- NFCorpus labels are noisy; SciFact (expert labels) is not run yet.

## Next steps
1. Add a false-premise rule to the grounded prompt; re-run and check `false_abstention_rate`.
2. MIRAGE with a PubMed-based corpus, and MedCPT (medical) retriever and reranker models.
3. SciFact; LLM query rewriting for short or vague queries (e.g. "Fish Fog").
4. Abstention experiment: MIRAGE questions against the Gale book only (coverage vs unsupported answers).

## Reproduce
```bash
python scripts/run_eval.py --config configs/v1_reproduction.yaml
python scripts/run_eval.py --config configs/improved.yaml
bash scripts/run_rerank_ablation.sh
python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/hybrid.yaml
python scripts/run_mirage.py --config configs/mirage/cot.yaml --per-dataset 100
python scripts/run_mirage.py --config configs/mirage/rag_bm25.yaml --per-dataset 100
python scripts/compare_runs.py results/<run A> results/<run B> --metrics <metrics>
```
Details: `README.md`.

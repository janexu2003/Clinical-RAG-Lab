# Evaluation report: offline_improved

35 questions (30 in scope). Means with bootstrap 95% CI.

| metric | mean | 95% CI | n |
|---|---|---|---|
| hit@1 | 0.633 | 0.467 – 0.8 | 30 |
| hit@3 | 0.767 | 0.6 – 0.9 | 30 |
| hit@5 | 0.967 | 0.9 – 1.0 | 30 |
| recall@5 | 0.95 | 0.867 – 1.0 | 30 |
| mrr | 0.745 | 0.617 – 0.864 | 30 |
| ndcg@5 | 0.79 | 0.679 – 0.891 | 30 |
| context_hit | 0.967 | 0.9 – 1.0 | 30 |
| context_chars | 5883.886 | 5219.771 – 6587.971 | 35 |
| latency_s | 0.007 | 0.005 – 0.008 | 35 |

## By question category

| category | n | hit@5 | mrr | context_hit |
|---|---|---|---|---|
| textbook_wording | 15 | 0.933 | 0.657 | 0.933 |
| lay_wording | 6 | 1.0 | 0.667 | 1.0 |
| numeric | 4 | 1.0 | 0.875 | 1.0 |
| false_premise | 2 | 1.0 | 1.0 | 1.0 |
| multi_part | 2 | 1.0 | 1.0 | 1.0 |
| reasoning | 1 | 1.0 | 1.0 | 1.0 |

## Failure causes

| cause | count | what to do |
|---|---|---|
| RETRIEVED_OK | 29 | Retrieval is fine. If the answer is wrong, fix generation: prompt, context order, context size. |
| OOS_NOT_EVALUATED | 5 | Out-of-scope question; run with an LLM to check abstention. |
| RANKING_MISS | 1 | Retrieved but ranked too low: add a reranker (cross-encoder or LLM) or section boost, or raise k. |

## Questions that need attention

**orig-13** (textbook_wording) What are the major causes of anemia?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 9} · candidate rank: 8 · final rank: None  
top retrieved: ['Anemias|Prevention|p198', 'Aplastic anemia|Causes and symptoms|p349', 'Aplastic anemia|Causes and symptoms|p350']  

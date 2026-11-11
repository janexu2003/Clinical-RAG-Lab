# Evaluation report: offline_baseline

35 questions (30 in scope). Means with bootstrap 95% CI.

| metric | mean | 95% CI | n |
|---|---|---|---|
| hit@1 | 0.567 | 0.4 – 0.733 | 30 |
| hit@3 | 0.767 | 0.6 – 0.9 | 30 |
| hit@5 | 0.833 | 0.7 – 0.967 | 30 |
| recall@5 | 0.817 | 0.683 – 0.933 | 30 |
| mrr | 0.677 | 0.539 – 0.811 | 30 |
| ndcg@5 | 0.69 | 0.553 – 0.82 | 30 |
| context_hit | 0.833 | 0.7 – 0.967 | 30 |
| context_chars | 2403.514 | 2362.971 – 2439.429 | 35 |
| latency_s | 0.011 | 0.009 – 0.013 | 35 |

## By question category

| category | n | hit@5 | mrr | context_hit |
|---|---|---|---|---|
| textbook_wording | 15 | 0.8 | 0.619 | 0.8 |
| lay_wording | 6 | 1.0 | 0.631 | 1.0 |
| numeric | 4 | 0.5 | 0.562 | 0.5 |
| false_premise | 2 | 1.0 | 1.0 | 1.0 |
| multi_part | 2 | 1.0 | 1.0 | 1.0 |
| reasoning | 1 | 1.0 | 1.0 | 1.0 |

## Failure causes

| cause | count | what to do |
|---|---|---|
| RETRIEVED_OK | 25 | Retrieval is fine. If the answer is wrong, fix generation: prompt, context order, context size. |
| RANKING_MISS | 5 | Retrieved but ranked too low: add a reranker (cross-encoder or LLM) or section boost, or raise k. |
| OOS_NOT_EVALUATED | 5 | Out-of-scope question; run with an LLM to check abstention. |

## Questions that need attention

**orig-04** (textbook_wording) What is an abscess?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 25} · candidate rank: 25 · final rank: None  
top retrieved: ['||p27', '||p27', '||p27']  

**orig-09** (textbook_wording) How is acne diagnosed?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 7} · candidate rank: 7 · final rank: None  
top retrieved: ['||p37', '||p37', '||p37']  

**orig-14** (textbook_wording) What are common symptoms of anemia?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 10} · candidate rank: 10 · final rank: None  
top retrieved: ['||p193', '||p197', '||p193']  

**new-07** (numeric) How long are oral antibiotics taken for acne?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 8} · candidate rank: 8 · final rank: None  
top retrieved: ['||p39', '||p39', '||p37']  

**new-10** (numeric) How long does hospital treatment for anorexia nervosa usually take?  
cause: `RANKING_MISS` · per-retriever rank of gold: {'bm25': 8} · candidate rank: 8 · final rank: None  
top retrieved: ['||p225', '||p224', '||p224']  

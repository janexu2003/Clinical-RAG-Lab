# clinical-rag-lab

A small, inspectable RAG lab for **one medical textbook** (Gale Encyclopedia of Medicine, vol. 1, ≈3,800 chunks).
It is built around three questions:

1. **The answer is wrong because retrieval is poor. What should I do?** → [diagnose, then fix by cause](#1-poor-retrieval-diagnose-first-then-fix-by-cause)
2. **The model hallucinates. What can I do?** → [prevent, detect, repair](#2-hallucination-prevent-detect-repair)
3. **How do I evaluate the system?** → [eval set, metrics, statistics](#3-evaluation)
4. **Does it hold beyond one textbook?** → [NFCorpus, SciFact, TREC-COVID, MIRAGE](#4-validate-on-established-benchmarks)


## 1. Poor retrieval: diagnose first, then fix by cause

"The answer is wrong" has several causes, and each has a different fix. Don't guess: trace the correct chunk through the pipeline. `ragkit/eval/diagnose.py` does this for every question:

| Diagnosis | What happened | Fix (config switch) |
|---|---|---|
| `NOT_IN_CORPUS` | No chunk covers the gold page | Data: add the source, fix PDF parsing, or the gold label is wrong |
| `RECALL_MISS` | No retriever has the gold chunk in its top 200 | Vocabulary mismatch: `query.transforms: [rewrite]`, `[multi_query]` or `[hyde]`; contextual headers; a better or medical embedding model |
| `CANDIDATE_MISS` | Some retriever found it, but it was lost in fusion | Add that retriever to the hybrid (`retrievers: [bm25, dense]`), raise `fetch_k`, tune `weights` |
| `RANKING_MISS` | It is in the candidate list but not in the top-k | Reranker: `rerank.type: cross_encoder` or `llm`, `section_boost: true`, or a larger `k` |
| `RETRIEVED_OK` | The gold chunk *was* in the context, but the answer is still wrong | Not a retrieval problem: fix the prompt, context order (`order: sandwich`) or context size |

The report also shows the gold chunk's rank **per retriever** (e.g. `{'bm25': 2, 'dense': 140}`), so you can see when BM25 and embeddings complement each other.


**End to end: `v1_reproduction` vs `improved`** (gpt-4o-mini; judge and claim checker are also gpt-4o-mini)

| Step | `v1_reproduction` (original chatbot) | `improved` |
|---|---|---|
| **Chunking** | `fixed`: 500-character windows per page, 20 overlap. Page footers stay in; chunks don't know which disease they belong to. | `section`: footers and captions removed; split by encyclopedia entry and section (Definition, Diagnosis, Treatment…); about 1,000 characters with 150 overlap; each chunk starts with a header like "Acne \| Diagnosis". |
| **Search** | `dense` only (MiniLM embeddings) | `bm25` + `dense` (hybrid): keyword and meaning search, merged |
| **Candidates / top-k** | fetch 3, use 3 | fetch 30, use the best 5 |
| **Reranking** | none | cross-encoder rescores the 30 candidates; section boost on |
| **Retrieval self-check** | none | `corrective`: the LLM judges whether the retrieved chunks are sufficient, and if not, rewrites the query and searches once more |
| **Context sent to the LLM** | the 3 chunks (about 1,400 characters) | the 5 best + their neighbouring chunks (about 5,400 characters) |
| **Prompt** | `v1_original`: "use the context… if you don't know, say so… three sentences max" | `grounded`: numbered sources, `[n]` citation on every sentence, answer the covered part, a fixed sentence for "not covered", no invented numbers |
| **Hallucination checks** | run, but only measure (`repair: false`) | run, and repair: rewrite the answer once, then remove sentences that are still unsupported |
| **Shared** | same PDF, same 35 questions, same LLM (gpt-4o-mini) | same PDF, same 35 questions, same LLM (gpt-4o-mini) |


| Metric | v1 | improved | Difference (95% CI) |
|---|---|---|---|
| Hit@1 | 0.400 | 0.867 | **+0.467** [+0.267, +0.667] |
| MRR | 0.583 | 0.933 | **+0.350** [+0.211, +0.494] |
| Faithfulness | 0.876 | 1.000 | **+0.124** [+0.056, +0.204] |
| Correctness | 0.933 | 0.917 | −0.017 [−0.133, +0.100] (no real difference) |
| False abstention (in scope) | 0.000 | 0.067 (2 questions) | |
| Out-of-scope correctly declined | 5/5 | 5/5 | |
| LLM calls / latency per question | 1.9 / 4.7 s | 3.3 / 8.1 s | |

What this shows:
- **Retrieval and grounding improved clearly.** The right chunk now comes first far more often, and every claim in the answers is supported by the sources (v1 included about 12% unsupported claims).
- **Answer correctness did not improve.** These 30 questions are easy enough that v1 already scored 0.93, so there is little room to improve.
- **The `grounded` prompt over-abstains on false-premise questions.** For "Which foods cause acne?" and "What important function does the appendix perform?", the right chunk was ranked 1st, but the model replied "The provided documents do not cover this." The question assumes something the sources contradict, and rule 4 of the prompt ("if the sources do not address the question, reply exactly …") was applied instead of correcting the premise. v1 answered both correctly. Next fix: add a prompt rule along the lines of "if the question assumes something the sources contradict, say so and give what the sources say", then re-run and check `false_abstention_rate`.


## 2. Hallucination: prevent, detect, repair

In RAG, a hallucination is **any statement the retrieved sources don't support**, even if it is medically true. For clinical use, unsupported means unsafe.

### Prevent (`generation.prompt`)

| Prompt | Idea |
|---|---|
| `v1_original` | Your original prompt: "three sentences max; if you don't know, say you don't know". Kept as the baseline. |
| `grounded` | Numbered sources; every sentence must cite `[n]`; answer the supported part and name the missing part; a fixed abstention phrase; never add numbers or doses that aren't in the sources. |
| `quote_then_answer` | The model first copies the exact sentences it relies on, then answers only from those quotes. Quotes can be **checked with string matching**, which is cheap and hard to fool. |

Other prevention knobs:
- `temperature: 0`
- fewer, better chunks: with a reranker, `k=5` beats `k=10` of noise
- `corrective.enabled`: the LLM grades whether the context is sufficient and searches again before answering

### Detect (`hallucination.checks`)

| Layer | Cost | What it catches |
|---|---|---|
| `numbers` | free | Numbers and doses absent from the sources. It is unit-aware: "20 weeks" is not supported by "20%", and "four to five months" counts as 4 and 5 months. |
| `citations` | free | Sentences without `[n]`, and citations to sources that don't exist |
| `lexical` | free | Sentences whose content words mostly don't appear in the cited sources |
| `quotes` | free | Quotes that are not exact text from any source, or are attributed to the wrong source |
| `llm_claims` | 1 LLM call | Splits the answer into atomic claims and labels each supported / not_found / contradicted. **Faithfulness** = share supported. |

`python scripts/demo_hallucination_checks.py` runs the free layers on real retrieved chunks:

```
=== invented dose ===
Isotretinoin is given at 0.5 mg per kg per day for 20 weeks [1].
  FLAG number_not_in_sources  0.5 mg
  FLAG number_not_in_sources  20 week
  FLAG low_lexical_support    (support=0.2)
=== fabricated quote + bad citation ===
Isotretinoin cures acne permanently in every patient [9]. Doctors recommend avoiding all dairy.
  FLAG invalid_citation 9 · uncited_sentence · low_lexical_support · fabricated_quote
```

What the free checks **cannot** catch: "Isotretinoin cures acne permanently in every patient" reuses the source's own words, so only its *meaning* is wrong. That needs `llm_claims`.

### Repair (`hallucination.repair: true`)

1. Regenerate once, giving the model the list of unsupported claims and invalid numbers or quotes.
2. If claims are still unsupported, delete those sentences and tell the user something was removed. If nothing remains, abstain.

### Measure
- `faithfulness` (claim support rate)
- `n_flags`
- `citation_rate`
- abstention on out-of-scope questions (`oos_correct_abstention`); answering those is pure hallucination by definition
- the `false_premise` category (e.g. "Which foods cause acne?" when the book says *no foods cause acne*), which tests whether the model goes along with the question

---

## 3. Evaluation

### a) The eval set (`eval_data/eval_set.json`, 35 questions)

| Category | n | What it tests |
|---|---|---|
| `textbook_wording` | 15 | Your original notebook questions |
| `lay_wording` | 6 | Vocabulary mismatch ("lazy eye", "bump at the base of the big toe", "wet the bed") |
| `numeric` | 4 | Durations and percentages: the riskiest place to hallucinate |
| `multi_part` | 2 | Two questions in one; needs evidence from two sections |
| `false_premise` | 2 | The question assumes something the book contradicts |
| `reasoning` | 1 | "Why" questions |
| `out_of_scope` | 5 | Topics absent from a 2004 book (COVID-19, semaglutide, mpox…); the correct behaviour is to abstain |

Every hand-written label was checked automatically: a key phrase from the answer must appear on the gold page in the raw PDF. That check caught one wrong page. The new questions were written from the book text by Claude, so **review them**.

To grow the set: `scripts/build_eval_set.py` writes LLM-generated questions in three personas (student, nurse, patient), sampled across entries, into a CSV. You review it and keep the good rows. Aim for 100+ questions.

### b) Metrics

| Level | Metric | Needs |
|---|---|---|
| Retrieval | Hit@k, Recall@k, MRR, nDCG@k on the ranked list; `context_hit` = is the gold page anywhere in what the LLM sees | nothing |
| Answer | `correctness`: an LLM judge compares the answer with the reference (1 / 0.5 / 0) | judge LLM |
| Grounding | `faithfulness`, `n_flags`, `citation_rate` | LLM for claims |
| Behaviour | `false_abstention_rate` (in scope, should be low), `oos_correct_abstention` (should be high) | LLM |
| Cost | `llm_calls`, `context_chars`, `latency_s` | nothing |

Always look at **retrieval and generation separately**. If `context_hit` is high but `correctness` is low, the problem is generation; the failure diagnosis does this split per question.


### c) Statistics: why 15 questions misled you

With *n* questions, one question is worth 100/*n* percentage points. Your notebook's Hit@10 of 0.60 on 15 questions has a 95% CI of roughly ±0.25.

- `run_eval.py` reports a **bootstrap 95% CI** for every metric.
- `compare_runs.py` runs a **paired bootstrap**: same questions, both configs. Paired comparisons are much more sensitive than comparing two separate means. It also reports wins/losses/ties and a permutation p-value.
- Rule: **if the CI of the difference includes 0, you have not shown an improvement.** Either collect more questions or report it as a trend, as the offline table above does.

### d) Can you trust the LLM judge?

`scripts/judge_agreement.py export results/<run>` exports 20 answers for you to label yourself. Then `score` computes Cohen's κ between your labels and the judge's. Above 0.6 is substantial agreement. Below that, fix the judge prompt or use a stronger judge model. Use a judge that is stronger than, or different from, the generator, so the model isn't marking its own homework.

### e) Workflow

1. Freeze the eval set, then run a baseline (`v1_reproduction`).
2. Read `report.md`. The failure causes tell you which stage to work on.
3. Change **one** setting, re-run, and `compare_runs.py` against the previous run.
4. Keep a change only if it helps the metric you care about without hurting faithfulness or abstention.
5. Re-check judge agreement whenever you change the judge or the prompt.
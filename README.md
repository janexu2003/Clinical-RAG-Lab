# clinical-rag-lab

A small, inspectable RAG lab for **one medical textbook** (Gale Encyclopedia of Medicine, vol. 1, ≈3,800 chunks).
It is built around three questions:

1. **The answer is wrong because retrieval is poor. What should I do?** → [diagnose, then fix by cause](#1-poor-retrieval-diagnose-first-then-fix-by-cause)
2. **The model hallucinates. What can I do?** → [prevent, detect, repair](#2-hallucination-prevent-detect-repair)
3. **How do I evaluate the system?** → [eval set, metrics, statistics](#3-evaluation)
4. **Does it hold beyond one textbook?** → [NFCorpus, SciFact, TREC-COVID, MIRAGE](#4-validate-on-established-benchmarks)

Everything is configuration-driven (`configs/*.yaml`), so each idea is one switch you can measure.
There is no vector database: a small corpus fits in memory. Every stage records what it did, so you can see *why* a question failed.

---

## Quickstart

```bash
pip install -r requirements.txt
cp /path/to/Medical_book.pdf data/                  # the PDF from your chatbot repo
pytest -q                                           # 21 offline tests

# Runs offline (no API key, no model download)
python scripts/run_eval.py --config configs/offline_baseline.yaml --retrieval-only
python scripts/run_eval.py --config configs/offline_improved.yaml --retrieval-only
python scripts/compare_runs.py results/offline_baseline_retrieval results/offline_improved_retrieval
python scripts/demo_hallucination_checks.py
python scripts/ask.py --config configs/offline_improved.yaml --retrieval-only "What is the main treatment for a lazy eye?"

# Full pipeline (embeddings + reranker + LLM)
echo "OPENAI_API_KEY=sk-..." > .env
python scripts/run_eval.py --config configs/v1_reproduction.yaml    # your original chatbot
python scripts/run_eval.py --config configs/improved.yaml
python scripts/compare_runs.py results/v1_reproduction results/improved

# Fully local / private (no data leaves the machine)
ollama pull llama3.1:8b && python scripts/run_eval.py --config configs/local_private.yaml
```

Each run writes `results/<name>/report.md` (metrics with 95% CIs, failure causes, and every question that needs attention), plus `per_question.jsonl` and `summary.json`.

---

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

**Order of fixes for a small corpus**, cheapest and most effective first:

1. **Look at your chunks.** Run `scripts/inspect_chunks.py`. In this book it shows 134 chunks under 120 characters, 91 of which are page-footer debris ("GALE ENCYCLOPEDIA OF MEDICINE 2 26 / Acne / GEM - 0001…").
2. **Chunk by structure.** Use entry/section chunks instead of fixed 500-character windows, and drop bibliography sections.
3. **Add a contextual header** to each chunk ("Acne | Diagnosis"), so a chunk about diagnosis always names its disease.
4. **Use hybrid search** (BM25 + dense). BM25 catches rare exact terms such as drug names and "acromegaly"; dense search catches paraphrases.
5. **Rerank the top 30** with a cross-encoder. It reads the question and passage together, which is much more accurate than comparing embeddings.
6. **Transform the query** when users use lay terms: "lazy eye" → "amblyopia".
7. **Expand neighbours**: add the chunk before and after, because answers often straddle a chunk boundary.

### Worked example: the lay-term problem

`python scripts/ask.py --config configs/offline_improved.yaml --retrieval-only "What is the main treatment for a lazy eye?"`

```
{'retrieval': 1, 'section_boost': 1}  Amblyopia | Description | p154   "Lazy eye is a common non-medical term ... amblyopia"
{'retrieval': 2, 'section_boost': 2}  Antibiotics, ophthalmic | Side effects
{'retrieval': 5, 'section_boost': 3}  Bell's palsy | Treatment
{'retrieval': 6, 'section_boost': 4}  Amblyopia | Treatment | p155      "The primary treatment for amblyopia is occlusion therapy..."
```

"Lazy eye" appears once in the whole book. BM25 finds that one sentence but ranks the actual treatment chunk 6th, and the section boost only lifts it to 4th. This is a `RANKING_MISS` close to becoming a `RECALL_MISS`. The fixes are a query rewrite (the LLM turns it into "amblyopia occlusion treatment") or a reranker.

### Offline results (BM25, 30 in-scope questions, 95% bootstrap CI)

| Step (one change at a time, `configs/ablations/`) | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| a1 fixed 500-char chunks (your v1 chunking) | 0.567 [0.40, 0.73] | 0.833 [0.70, 0.97] | 0.677 | 0.690 |
| a2 + footer removal | 0.533 | 0.800 | 0.652 | 0.656 |
| a3 section chunks, no header | 0.600 | 0.833 | 0.708 | 0.723 |
| a4 + contextual header | 0.633 | 0.900 | 0.736 | 0.759 |
| a5 + section-intent boost | 0.633 | 0.967 [0.90, 1.00] | 0.745 | 0.790 |
| a6 + neighbour expansion | 0.633 | 0.967 | 0.745 | 0.790 |
| a7 + LSA hybrid (offline stand-in for dense) | 0.633 | 0.967 | 0.762 | 0.808 |

Paired comparison, a1 → a6 (`compare_runs.py`): Hit@5 +0.13, 95% CI [0.00, +0.30], 5 questions better and 1 worse, p = 0.22.
**The direction is consistent, but 30 questions are not enough to call it proven.** See [evaluation](#3-evaluation).

Two more points:
- Neighbour expansion doesn't change retrieval metrics, because the gold page is already found. It adds about 70% more context (2.4k → 5.9k characters) so the LLM sees complete passages. Its value appears in *answer* correctness, which needs an LLM run to measure.
- The table above uses BM25 only. The results with the full stack (MiniLM dense search, hybrid, cross-encoder) are below.

### Results with the full stack (35 questions, run locally on a MacBook Air)

**Reranker ablation** (`configs/rerank_ablation/`, `scripts/run_rerank_ablation.sh`). All four configs use the same section chunks and BM25 + dense hybrid search; only the reranking step changes. Retrieval only, so no API calls.

| Config | Hit@1 | Hit@5 | MRR | nDCG@5 | Time per question |
|---|---|---|---|---|---|
| r0 no reranking | 0.700 | 1.000 | 0.817 | 0.836 | 0.03 s |
| r1 section boost only | 0.700 | 1.000 | 0.822 | 0.842 | 0.02 s |
| r2 cross-encoder only | **0.867** | 1.000 | **0.933** | 0.943 | 0.13 s |
| r3 cross-encoder + section boost | **0.867** | 1.000 | **0.933** | **0.948** | 0.13 s |

Paired comparisons (`compare_runs.py`, 30 in-scope questions):
- **The cross-encoder helps** (r0 → r2): MRR +0.117, 95% CI [+0.022, +0.222]; nDCG@5 +0.108, CI [+0.035, +0.187]; 10 questions better, 1 worse. These CIs exclude 0, so this gain holds up statistically. Hit@1 +0.167 (CI [0.000, +0.333]) is borderline. It costs about 0.1 s per question on a laptop CPU.
- **The section boost adds nothing measurable here.** r0 → r1: nDCG@5 +0.007; r2 → r3: +0.005. With BM25 alone it helped (a4 → a5 above), but once dense search and the cross-encoder are in place, they already put the right section first.
- **Chunking + hybrid search** (BM25 on fixed chunks → r0): Hit@5 +0.167, CI [+0.033, +0.300]. Five questions went from `RANKING_MISS` to retrieved.

**End to end: `v1_reproduction` vs `improved`** (gpt-4o-mini; judge and claim checker are also gpt-4o-mini)

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

---

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

---

## 4. Validate on established benchmarks

The textbook eval set is small (30 in-scope questions) and was written by us. Established benchmarks give you more questions, which means narrower confidence intervals, and labels you didn't write, so your results can be compared with published ones.

| Benchmark | Measures | Size | Runner |
|---|---|---|---|
| **NFCorpus** (BEIR) | Retrieval: nDCG@10, Recall@100, MRR@10 | ≈3.6k docs, 323 test queries, graded labels | `run_beir.py` |
| **SciFact** (BEIR) | Retrieval | ≈5.2k abstracts, 300 test claims | `run_beir.py` |
| **TREC-COVID** (BEIR) | Retrieval on a larger corpus | ≈171k docs, 50 queries | `run_beir.py` |
| **MIRAGE** + MedRAG Textbooks | Answer accuracy with and without RAG | 7,663 multiple-choice questions; ≈126k snippets | `run_mirage.py` |

```bash
# 0. get the data (on your own machine; needs internet)
python scripts/download_benchmarks.py beir nfcorpus scifact
python scripts/download_benchmarks.py mirage
pip install huggingface_hub && python scripts/download_benchmarks.py medrag-textbooks

# 1. retrieval: one change at a time, then paired comparison
python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/bm25.yaml
python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/hybrid.yaml
python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/hybrid_rerank.yaml
python scripts/compare_runs.py results/beir_nfcorpus_bm25 results/beir_nfcorpus_hybrid_rerank \
    --metrics ndcg@10 recall@100 mrr@10
python scripts/compare_runs.py results/beir_nfcorpus_bm25 results/beir_nfcorpus_hybrid \
    --metrics ndcg@10 recall@100 mrr@10
python scripts/compare_runs.py results/beir_nfcorpus_hybrid results/beir_nfcorpus_hybrid_rerank \
    --metrics ndcg@10 recall@100 mrr@10

# 2. answers: the same 500 questions (100 per dataset, fixed seed) for every config
python scripts/run_mirage.py --config configs/mirage/cot.yaml --per-dataset 100
python scripts/run_mirage.py --config configs/mirage/rag_bm25.yaml --per-dataset 100
python scripts/compare_runs.py results/mirage_cot results/mirage_rag_bm25 --metrics correct
```

**Notes**
- **Metrics match the standard.** BEIR metrics follow `trec_eval` (graded nDCG@10, Recall@100). The implementation was checked against `pytrec_eval` on 300 random rankings and matches exactly, so your numbers are comparable with BEIR's published BM25 and dense baselines.
- **The runners report extra detail.** `run_beir.py` also runs the failure diagnosis on every query with nDCG@10 = 0 (`RECALL_MISS` / `CANDIDATE_MISS` / `RANKING_MISS`).
- **MIRAGE uses question-only retrieval by default**, as MIRAGE recommends; `--retrieve-with-options` changes that. Runs resume if interrupted.
- **Smoke-test before spending money.** `--limit 20` (BEIR) or `--per-dataset 5` (MIRAGE), and `generation: {llm: dummy}` tests the plumbing for free.
- **Cost and time:**
  - BM25 configs index in seconds to minutes.
  - Dense configs embed the whole corpus on CPU the first time and are cached in `.cache/` afterwards. Fine for NFCorpus and SciFact; slow for TREC-COVID and the 126k Textbooks.
  - MIRAGE makes 1 LLM call per question, plus 1 per question with `rewrite`.
- **What carries over from the textbook work:** hybrid search, reranking, query rewriting, the corrective loop and the statistics. The Gale-specific cleaning and section boost don't apply; there, the equivalent is using the document title as a header, which the BEIR loader does.
- **How to report:** your config next to the published baseline (BEIR paper / MedRAG paper Table 6), with a 95% CI and a paired comparison against your own baseline run.

---

## Suggested experiments (with an API key; each run is roughly 4–6 LLM calls per question, cheap on a small model such as gpt-4o-mini)

| # | Compare | Question it answers |
|---|---|---|
| 1 | `v1_reproduction` vs `improved` | How much better is the whole system? |
| 2 | `improved` with `retrievers: [dense]` / `[bm25]` / `[bm25, dense]` | Does hybrid search help on lay versus textbook wording? |
| 3 | `rerank.type: none` / `cross_encoder` / `llm` | Is a reranker worth the latency? |
| 4 | `query.transforms: []` / `[rewrite]` / `[multi_query]` / `[hyde]` | Which fixes lay-term questions best, and at what cost? |
| 5 | `prompt: v1_original` / `grounded` / `quote_then_answer` | Which prompt hallucinates least without over-abstaining? |
| 6 | `hallucination.repair: false` vs `true` | How much does repair raise faithfulness, and does it hurt correctness? |
| 7 | `dense_model`: MiniLM vs `BAAI/bge-small-en-v1.5` vs MedCPT | Do domain embeddings help on a small corpus? |
| 8 | `local_private` vs `improved` | Privacy and cost versus quality (Llama 3.1 8B vs gpt-4o-mini) |

---

## Layout

```
ragkit/
  corpus.py         PDF -> cleaned pages -> fixed or section chunks (+ contextual headers)
  retrievers.py     bm25 | tfidf | lsa | dense | hybrid (RRF); per-retriever ranks for diagnosis
  query.py          rewrite | multi_query | decompose | hyde
  rerank.py         cross_encoder | llm | section_boost
  context.py        neighbour expansion, ordering ("sandwich"), character budget
  generate.py       prompts: v1_original | grounded | quote_then_answer
  hallucination.py  offline checks, LLM claim verification, repair
  pipeline.py       config -> pipeline; corrective grading loop; full traces
  llm.py            openai:<model> | ollama:<model> | stub (tests)
  eval/             dataset, metrics, diagnose, stats, runner (reports)
  benchmarks/       beir.py (BEIR loader + trec_eval-style metrics), mirage.py (MIRAGE + MedRAG corpus)
scripts/            run_eval, compare_runs, ask, inspect_chunks, build_eval_set,
                    judge_agreement, demo_hallucination_checks,
                    download_benchmarks, run_beir, run_mirage
configs/            offline_*, v1_reproduction, improved, quote_then_answer, local_private, ablations/,
                    beir/ (bm25 … hybrid_rerank_rewrite), mirage/ (cot, rag_bm25 …)
eval_data/          eval_set.json
```

## Limitations
- Page-level gold labels are coarse: a retrieved chunk can be on the right page but in the wrong paragraph. Answer-level judging covers this.
- The section-intent boost relies on hand-written rules tuned for encyclopedia-style sections. A reranker is the general solution.
- The LLM judge and claim verifier are LLMs too. Calibrate them (section 3d) before relying on their numbers.
- A 2004 encyclopedia is not a current clinical reference. This is a methods project, not a medical tool.

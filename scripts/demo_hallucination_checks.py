"""
Offline demo: what the free hallucination checks catch. No LLM needed.

Retrieves real chunks for an acne question, then runs the checks on three hand-written answers:
a faithful one, one with an invented dose, and one with a fabricated quote and a bad citation.

    python scripts/demo_hallucination_checks.py
"""
import _path  # noqa: F401

from ragkit.hallucination import offline_checks
from ragkit.pipeline import RAGPipeline, load_config

pipe = RAGPipeline(load_config("configs/improved.yaml", overrides={
    "generation": {"llm": "none"},        # no API calls
    "corrective": {"enabled": False},     # corrective grading needs an LLM
}))
q = "How long are oral antibiotics taken for acne, and what is used for severe acne?"
#ctx = pipe.retrieve(q).final
rt = pipe.retrieve(q)
ctx = rt.final
print("QUESTION:", q)
print("search queries:", rt.queries)            # shows any rewritten queries
print("corrective grading:", rt.notes)          # e.g. ['grade[0]=sufficient']

print(f"{len(ctx)} chunks retrieved")
for i, c in enumerate(ctx, 1):
    print(f"  [{i}] {c.cite_label()}")

src = next((i for i, c in enumerate(ctx, 1) if "two to four months" in c.text), None)
if src is None:
    raise SystemExit("The antibiotics chunk was not retrieved, so the 'faithful' answer has no "
                     "source to cite. That is itself a retrieval miss worth looking at.")
bad = len(ctx) + 1                        # a citation number that is always out of range

answers = {
    "faithful": (f"Oral antibiotics are taken daily for two to four months [{src}]. "
                 f"Severe acne with cysts and nodules is treated with oral isotretinoin (Accutane) [{src}].", None),
    "invented dose": (f"Oral antibiotics are taken daily for two to four months [{src}]. "
                      f"Isotretinoin is given at 0.5 mg per kg per day for 20 weeks [{src}].", None),
    "fabricated quote + bad citation": (
        "Isotretinoin cures acne permanently in every patient [{bad}]. Doctors recommend avoiding all dairy.",
        [{"source": src, "quote": "Isotretinoin cures acne permanently in all patients."}]),
}
for name, (ans, quotes) in answers.items():
    print(f"\n=== {name} ===\n{ans}")
    flags = offline_checks(ans, ctx, quotes)
    if not flags:
        print("  no flags")
    for f in flags:
        detail = f.get("value") or f.get("quote") or f.get("sentence", "")
        print(f"  FLAG {f['type']:22s} {str(detail)[:90]}" + (f"  (support={f['support']})" if "support" in f else ""))

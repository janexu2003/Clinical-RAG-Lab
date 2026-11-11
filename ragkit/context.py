"""
Context assembly: what exactly the LLM sees.

* neighbours - add the chunk before/after from the same section, because answers are often
               split across a chunk boundary ("...not difficult to | diagnose. The doctor...")
* ordering   - LLMs read the start and end of a long context best ("lost in the middle",
               Liu et al. 2023). "sandwich" puts the best chunks first and last.
* budget     - cap total characters so cost and distraction stay bounded
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ragkit.types import Chunk


class NeighbourIndex:
    def __init__(self, chunks: List[Chunk]):
        self.by_block: Dict[Tuple, Dict[int, Chunk]] = defaultdict(dict)
        for c in chunks:
            self.by_block[(c.source, c.entry, c.section, c.page if not c.entry else None)][c.chunk_index] = c

    def neighbours(self, c: Chunk, window: int = 1) -> List[Chunk]:
        block = self.by_block.get((c.source, c.entry, c.section, c.page if not c.entry else None), {})
        return [block[i] for i in range(c.chunk_index - window, c.chunk_index + window + 1)
                if i != c.chunk_index and i in block]


def assemble(ranked: List[Chunk], k: int, nidx: NeighbourIndex = None, window: int = 0,
             order: str = "relevance", max_chars: int = 12000) -> List[Chunk]:
    """Top-k hits, each expanded into a group [prev neighbour, hit, next neighbour]."""
    groups: List[List[Chunk]] = []
    seen = set()
    for c in ranked[:k]:
        if c.id in seen:
            continue
        group = [c]
        if window and nidx is not None:
            nb = nidx.neighbours(c, window)
            group = sorted([c] + nb, key=lambda x: x.chunk_index)   # keep reading order
        group = [g for g in group if g.id not in seen]
        seen.update(g.id for g in group)
        if group:
            groups.append(group)

    if order == "sandwich" and len(groups) > 2:
        # best, 3rd, 5th, ..., 4th, 2nd -> strongest evidence at both ends of the prompt
        groups = groups[0::2] + groups[1::2][::-1]

    out, total = [], 0
    for g in groups:
        size = sum(len(c.text) for c in g)
        if out and total + size > max_chars:
            continue
        out.extend(g)
        total += size
    return out

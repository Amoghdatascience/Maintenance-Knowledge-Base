"""
chunking.py

Structure-aware chunking for the cluster markdown documents.

Why not fixed 1000/200 character windows
-----------------------------------------
1. The documents already have meaningful boundaries: "## Common problem
   terms", "## Inclusion guidance", "### Example 3". A fixed window cuts
   straight through them, so one chunk ends mid-example and the next starts
   with half a record and no idea which cluster it belongs to.
2. all-MiniLM-L6-v2 embeds at most 256 word-piece tokens and silently drops
   the rest. This text is dense: "- 7200.0: 288" or "N559ND: 27" is ~2.5-3
   characters per token, so a 1000-character chunk can be ~330-400 tokens and
   lose its last third before it is ever embedded.
3. A chunk cut from the middle of a document has no cluster name in it, so
   its embedding does not know what it is about.

What this does instead
----------------------
- Splits on "## " sections, and inside a section on "### " sub-blocks
  (each example record is one unit).
- Packs whole units into chunks up to chunk_size characters, never splitting
  a unit that fits.
- Prefixes every chunk with "Cluster c_22: engine idle rpm issue | Example
  records", so every chunk is self-describing for both embeddings and BM25.
- Only a unit longer than chunk_size is cut, on line boundaries, with
  chunk_overlap characters (whole lines) carried over so an entry that spans
  the cut appears complete in at least one chunk.
- Files with no "## " headers fall back to the recursive character splitter.

No overlap is added between units: they are self-contained, so overlap would
only duplicate text, spend the evidence budget twice, and make BM25 count the
same terms twice.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


Chunk = Tuple[str, Dict[str, str]]


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()

    return fallback


def _sections(text: str) -> List[Tuple[str, str]]:
    """
    Split a markdown document into (section name, body) pairs on "## ".

    Text before the first "## " (minus the "# " title) becomes "Overview".
    """
    sections: List[Tuple[str, str]] = []
    name = "Overview"
    lines: List[str] = []

    for line in text.splitlines():
        if line.startswith("# ") and not sections and not lines:
            continue

        if line.startswith("## "):
            if "\n".join(lines).strip():
                sections.append((name, "\n".join(lines).strip()))
            name = line[3:].strip()
            lines = []
            continue

        lines.append(line)

    if "\n".join(lines).strip():
        sections.append((name, "\n".join(lines).strip()))

    return sections


def _units(body: str) -> List[str]:
    """
    Split a section body into indivisible units on "### " headers.
    """
    parts = re.split(r"(?m)^(?=### )", body)
    return [part.strip() for part in parts if part.strip()]


def _split_long_unit(unit: str, limit: int, overlap: int) -> List[str]:
    """
    Cut an oversized unit on line boundaries, carrying whole trailing lines
    (up to `overlap` characters) into the next piece.
    """
    lines = unit.splitlines()
    pieces: List[str] = []
    current: List[str] = []

    for line in lines:
        # A single line longer than the limit is hard-wrapped.
        while len(line) > limit:
            pieces.append(line[:limit])
            line = line[limit - overlap:] if overlap < limit else line[limit:]

        candidate = "\n".join(current + [line])

        if current and len(candidate) > limit:
            pieces.append("\n".join(current))

            carried: List[str] = []
            size = 0

            for previous in reversed(current):
                if size + len(previous) + 1 > overlap:
                    break
                carried.insert(0, previous)
                size += len(previous) + 1

            current = carried + [line]
        else:
            current.append(line)

    if current:
        pieces.append("\n".join(current))

    return [piece for piece in pieces if piece.strip()]


def split_markdown(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    fallback_title: str = "",
) -> List[Chunk]:
    """
    Chunk one markdown document. Returns (chunk text, extra metadata) pairs.

    chunk_size bounds the WHOLE chunk, header line included.
    """
    title = _title_of(text, fallback_title)

    # Header length budget: title plus a generous allowance for section names,
    # so packed chunks stay under chunk_size once the header is attached.
    room = max(150, chunk_size - len(title) - 60)
    overlap = min(chunk_overlap, room // 2)

    # Flatten into (section, unit) pairs. The first unit of each section keeps
    # its "## name" line, so a packed chunk still reads as markdown.
    units: List[Tuple[str, str]] = []

    for section, body in _sections(text):
        for position, unit in enumerate(_units(body)):
            labelled = f"## {section}\n{unit}" if position == 0 else unit

            if len(labelled) <= room:
                units.append((section, labelled))
            else:
                units.extend(
                    (section, piece)
                    for piece in _split_long_unit(labelled, room, overlap)
                )

    # Pack whole units greedily. Small neighbouring sections (Overview +
    # Purpose, problem terms + action terms) share a chunk instead of each
    # becoming a 150-character fragment with too little signal to retrieve.
    packed: List[Tuple[List[str], str]] = []
    names: List[str] = []
    current = ""

    for section, unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit

        if current and len(candidate) > room:
            packed.append((names, current))
            names, current = [section], unit
        else:
            current = candidate
            if section not in names:
                names.append(section)

    if current:
        packed.append((names, current))

    chunks: List[Chunk] = []

    for section_names, body_text in packed:
        label = ", ".join(section_names)
        header = f"{title} | {label}\n" if title else f"{label}\n"
        chunks.append((header + body_text, {"section": label}))

    return chunks


def is_markdown(text: str) -> bool:
    return bool(re.search(r"(?m)^## ", text))

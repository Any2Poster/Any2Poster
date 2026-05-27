"""Stage 1.5: CHUNK

Semantic-boundary-aware chunking of parsed sections.
- Sections under 800 tokens are kept whole (no split)
- Equations and algorithm blocks are atomic (never split mid-block)
- Topic shift detection via signal words for optimal split points
- Each chunk carries context_before/context_after for LLM grounding
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from any2poster.models import ChunkedDocument, TextChunk

if TYPE_CHECKING:
    from any2poster.models import ParsedDocument

TOPIC_SHIFT_SIGNALS = re.compile(
    r"(?:^|\n)(?:"
    r"however|in contrast|on the other hand|conversely|"
    r"next we|we then|we also|we further|"
    r"to evaluate|for evaluation|we evaluate|we compare|"
    r"specifically|in particular|more precisely|"
    r"as shown in|as illustrated|table \d|figure \d|fig\. \d|"
    r"our key|the main|the primary|importantly|notably"
    r")",
    re.IGNORECASE,
)

EQUATION_BLOCK = re.compile(
    r"(\$\$[\s\S]*?\$\$|\\begin\{(?:equation|align|gather|multline)\}[\s\S]*?"
    r"\\end\{(?:equation|align|gather|multline)\})",
    re.MULTILINE,
)

ALGORITHM_BLOCK = re.compile(
    r"(\\begin\{algorithm\}[\s\S]*?\\end\{algorithm\}|"
    r"```[\s\S]*?```)",
    re.MULTILINE,
)

MIN_CHUNK_TOKENS = 100
MAX_CHUNK_TOKENS = 800
CONTEXT_WINDOW_TOKENS = 50


def run_chunk_stage(parsed: "ParsedDocument") -> ChunkedDocument:
    all_chunks: list[TextChunk] = []
    chunk_counter = 0

    for section in parsed.sections:
        if section.section_type.value == "references":
            continue

        content = section.content.strip()
        if not content:
            continue

        token_count = len(content.split())

        if token_count <= MAX_CHUNK_TOKENS:
            chunk_counter += 1
            all_chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    section_id=section.section_id,
                    content=content,
                    token_count=token_count,
                    figure_ids=section.figure_ids.copy(),
                )
            )
            continue

        atomic_blocks = _mark_atomic_blocks(content)
        paragraphs = _split_to_paragraphs(content, atomic_blocks)

        section_chunks = _merge_paragraphs_into_chunks(
            paragraphs, section.section_id, section.figure_ids
        )

        for i, chunk in enumerate(section_chunks):
            chunk_counter += 1
            chunk.chunk_id = f"chunk_{chunk_counter}"

            if i > 0:
                prev_words = section_chunks[i - 1].content.split()
                chunk.context_before = " ".join(prev_words[-CONTEXT_WINDOW_TOKENS:])
            if i < len(section_chunks) - 1:
                next_words = section_chunks[i + 1].content.split()
                chunk.context_after = " ".join(next_words[:CONTEXT_WINDOW_TOKENS])

            all_chunks.append(chunk)

    if all_chunks and len(all_chunks[-1].content.split()) < MIN_CHUNK_TOKENS and len(all_chunks) > 1:
        last = all_chunks.pop()
        all_chunks[-1].content += "\n\n" + last.content
        all_chunks[-1].token_count = len(all_chunks[-1].content.split())
        all_chunks[-1].figure_ids.extend(last.figure_ids)

    return ChunkedDocument(
        chunks=all_chunks,
        total_chunks=len(all_chunks),
    )


def _mark_atomic_blocks(text: str) -> list[tuple[int, int]]:
    blocks = []
    for pattern in (EQUATION_BLOCK, ALGORITHM_BLOCK):
        for match in pattern.finditer(text):
            blocks.append((match.start(), match.end()))
    blocks.sort(key=lambda x: x[0])
    return blocks


def _is_inside_atomic(pos: int, atomic_blocks: list[tuple[int, int]]) -> bool:
    for start, end in atomic_blocks:
        if start <= pos <= end:
            return True
        if start > pos:
            break
    return False


def _split_to_paragraphs(
    text: str, atomic_blocks: list[tuple[int, int]]
) -> list[str]:
    raw_splits = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    buffer = ""

    char_pos = 0
    for part in raw_splits:
        part_start = text.find(part, char_pos)
        part_end = part_start + len(part) if part_start >= 0 else char_pos + len(part)

        if _is_inside_atomic(part_start, atomic_blocks):
            buffer += "\n\n" + part if buffer else part
        else:
            if buffer:
                buffer += "\n\n" + part
                paragraphs.append(buffer.strip())
                buffer = ""
            else:
                stripped = part.strip()
                if stripped:
                    paragraphs.append(stripped)

        char_pos = part_end

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs


def _find_best_split(paragraph: str) -> int | None:
    matches = list(TOPIC_SHIFT_SIGNALS.finditer(paragraph))
    if not matches:
        return None
    mid = len(paragraph) // 2
    best = min(matches, key=lambda m: abs(m.start() - mid))
    if best.start() < 50 or best.start() > len(paragraph) - 50:
        return None
    return best.start()


def _merge_paragraphs_into_chunks(
    paragraphs: list[str],
    section_id: str,
    figure_ids: list[str],
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    figure_assigned = False

    for para in paragraphs:
        para_tokens = len(para.split())

        if para_tokens > MAX_CHUNK_TOKENS:
            if current_parts:
                merged = "\n\n".join(current_parts)
                fids = figure_ids.copy() if not figure_assigned else []
                if fids:
                    figure_assigned = True
                chunks.append(
                    TextChunk(
                        chunk_id="",
                        section_id=section_id,
                        content=merged,
                        token_count=len(merged.split()),
                        figure_ids=fids,
                    )
                )
                current_parts = []
                current_tokens = 0

            split_pos = _find_best_split(para)
            if split_pos:
                part_a = para[:split_pos].strip()
                part_b = para[split_pos:].strip()
                for part in (part_a, part_b):
                    if part:
                        chunks.append(
                            TextChunk(
                                chunk_id="",
                                section_id=section_id,
                                content=part,
                                token_count=len(part.split()),
                                figure_ids=[],
                            )
                        )
            else:
                words = para.split()
                half = len(words) // 2
                chunks.append(
                    TextChunk(
                        chunk_id="",
                        section_id=section_id,
                        content=" ".join(words[:half]),
                        token_count=half,
                        figure_ids=[],
                    )
                )
                chunks.append(
                    TextChunk(
                        chunk_id="",
                        section_id=section_id,
                        content=" ".join(words[half:]),
                        token_count=len(words) - half,
                        figure_ids=[],
                    )
                )
            continue

        if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current_parts:
            merged = "\n\n".join(current_parts)
            fids = figure_ids.copy() if not figure_assigned else []
            if fids:
                figure_assigned = True
            chunks.append(
                TextChunk(
                    chunk_id="",
                    section_id=section_id,
                    content=merged,
                    token_count=len(merged.split()),
                    figure_ids=fids,
                )
            )
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        merged = "\n\n".join(current_parts)
        fids = figure_ids.copy() if not figure_assigned else []
        chunks.append(
            TextChunk(
                chunk_id="",
                section_id=section_id,
                content=merged,
                token_count=len(merged.split()),
                figure_ids=fids,
            )
        )

    return chunks
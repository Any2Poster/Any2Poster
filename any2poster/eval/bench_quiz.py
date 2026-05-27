"""BenchQuiz — any2poster Bench evaluation module.

Domain-neutral PaperQuiz-style evaluation for the any2poster Bench.
Designed to work across all 5 content genres (research, news, educational,
business, fiction) and all 8 input modalities.

Protocol
--------
1. Generate 10 verbatim + 10 interpretive MCQs from the *source* document,
   using genre-aware prompts with domain-neutral aspect categories.
2. Answer every question using *only* the poster image — no source access.
3. Score correct answers and compute raw accuracy + density-augmented score.

This module is fully independent of paperquiz.py — it shares no imports with
it and does not touch the Paper2Poster eval format in any way.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────────
# Supported genres
# ─────────────────────────────────────────────────────────────────────────────

GENRES = ("research", "news", "educational", "business", "fiction")

_GENRE_CONTEXT: dict[str, str] = {
    "research": (
        "The source is an academic or scientific paper, thesis, or technical report. "
        "Questions should target the research problem, methodology, experimental findings, "
        "quantitative results, and stated contributions or limitations."
    ),
    "news": (
        "The source is a news article, investigative report, or journalistic piece. "
        "Questions should target the core event or issue, the key people and organizations "
        "involved, what happened, when and where, and the reported consequences or context."
    ),
    "educational": (
        "The source is an educational or instructional document — a textbook chapter, tutorial, "
        "course notes, or how-to guide. Questions should target the concept or skill being "
        "taught, key definitions or terminology, the steps or process described, and the "
        "stated learning objectives or practical applications."
    ),
    "business": (
        "The source is a business or professional document — an earnings report, annual report, "
        "policy brief, product specification, investor presentation, or corporate white paper. "
        "Questions should target the organization and its context, key metrics, financial figures "
        "or policy positions described, stated strategic objectives, and reported outcomes or "
        "recommendations."
    ),
    "fiction": (
        "The source is a work of fiction or narrative non-fiction — a novel excerpt, short story, "
        "memoir passage, or narrative essay. Questions should target the main characters or "
        "narrator, the central conflict or narrative arc, the setting, key plot events or turning "
        "points, and the themes, tone, or resolution."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# QA generation prompts — verbatim (detail-level)
# ─────────────────────────────────────────────────────────────────────────────

VERBATIM_QA_SYSTEM = (
    "You are a Question-Generation agent for information-extraction evaluation. "
    "Your task is to read the supplied source document and produce exactly 20 multiple-choice "
    "questions that test whether a reader has retained specific, concrete details from the source. "
    "Every question must be answerable by locating an explicit phrase, statistic, name, date, "
    "or direct statement in the source text — do not require inference beyond what is written. "
    "Follow all instructions below precisely. "
    "CRITICAL JSON RULE: Never use the straight double-quote character (\") inside any "
    "question, option, or answer string value. Use single quotes or rephrase instead. "
    "Your entire response must be valid JSON with no unescaped double quotes inside strings."
)

_VERBATIM_INSTRUCTIONS = """\
Source document context: {genre_context}

Instructions:
1. Read the source document carefully with the above context in mind.
2. Produce exactly 20 factual, detail-oriented questions.
   - Each question must map to one specific, locatable fact, phrase, or value in the source.
   - Vary difficulty: mix prominent headline facts with specific supporting details.
   - No duplicate or near-duplicate questions.
3. Distribute the 20 questions across these domain-neutral aspects. Every aspect must appear
   at least twice; aim for 2-3 questions per aspect:
     A. Main topic / subject matter — what this document is fundamentally about
     B. Central argument, claim, or narrative — the core message being communicated
     C. Key entities — specific people, organizations, characters, or named concepts
     D. Important facts, data, statistics, dates, or specific events explicitly stated
     E. Process, methodology, sequence of steps, or how something works or unfolds
     F. Conclusions, outcomes, results, or resolution explicitly stated in the source
     G. Supporting details, specific examples, or illustrative cases cited
     H. Stated implications, significance, applications, or direct takeaways
4. Write three distractor options that are plausible but clearly wrong based on the source.
   Do not use trick questions. Each question must have exactly one unambiguously correct answer.
5. EXCLUDE: author acknowledgements, footnotes, legal disclaimers, reference lists, and any
   content that would not appear on a summary poster or slide.
6. Output only this exact JSON object with 20 entries — no commentary or extra text:
{{
  "Question 1": {{
    "aspect": "<A–H>",
    "question": "<single clear sentence ending in a question mark>",
    "options": ["A. <choice>", "B. <choice>", "C. <choice>", "D. <choice>"],
    "answer": "<Letter>. <exact correct option text>"
  }},
  "Question 2": {{ ... }},
  ...
  "Question 20": {{ ... }}
}}
7. Distribute correct answers roughly equally across A, B, C, and D — approximately 5 each.
Think step by step. Verify every question is grounded in explicit source text before finalising.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# QA generation prompts — interpretive (comprehension-level)
# ─────────────────────────────────────────────────────────────────────────────

INTERPRETIVE_QA_SYSTEM = (
    "You are a Question-Generation agent for comprehension evaluation. "
    "Your task is to read the supplied source document and produce exactly 20 multiple-choice "
    "questions that test high-level understanding — the central argument, overall narrative, "
    "core purpose, and primary takeaways. Every question must still be answerable by locating "
    "explicit content in the source; do not require inference beyond what is written. "
    "Follow all instructions below precisely. "
    "CRITICAL JSON RULE: Never use the straight double-quote character (\") inside any "
    "question, option, or answer string value. Use single quotes or rephrase instead. "
    "Your entire response must be valid JSON with no unescaped double quotes inside strings."
)

_INTERPRETIVE_INSTRUCTIONS = """\
Source document context: {genre_context}

Instructions:
1. Read the source document closely with the above context in mind.
2. Produce exactly 20 comprehension-level questions that probe big-picture understanding.
   - Target the overall argument, narrative arc, purpose, and significance — not granular details.
   - Avoid questions on specific numbers, minor named entities, or low-level procedural steps;
     those belong in the verbatim question set.
   - No duplicate or near-duplicate questions.
3. Cover each of the following aspects at least twice — aim for 2-3 questions per aspect:
     A. Main topic / subject matter — what this document is fundamentally about and why it exists
     B. Central argument, claim, or narrative — the overarching message or story
     C. The primary entities (people, organizations, characters) and their role in the overall work
     D. The significance or importance of the key findings, events, or claims described
     E. The overall approach, structure, strategy, or narrative method used in the document
     F. The primary conclusion, outcome, or resolution — the definitive "so what"
     G. The broader context, motivation, or background that frames the content
     H. The intended takeaway, lesson, application, or implication for the reader
4. Write three distractor options reflecting plausible but incorrect high-level interpretations.
   Each question must have exactly one unambiguously correct answer based on the source.
5. EXCLUDE: author acknowledgements, footnotes, legal disclaimers, reference lists, and granular
   minutiae that would not appear on a summary poster.
6. Output only this exact JSON object with 20 entries — no commentary or extra text:
{{
  "Question 1": {{
    "aspect": "<A–H>",
    "question": "<single clear sentence ending in a question mark>",
    "options": ["A. <choice>", "B. <choice>", "C. <choice>", "D. <choice>"],
    "answer": "<Letter>. <exact correct option text>"
  }},
  "Question 2": {{ ... }},
  ...
  "Question 20": {{ ... }}
}}
7. Distribute correct answers roughly equally across A, B, C, and D — approximately 5 each.
Think step by step. Ensure each question probes overall understanding and has a single
unambiguous answer grounded in the source text.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Poster answering prompt
# ─────────────────────────────────────────────────────────────────────────────

BENCH_ANSWER_SYSTEM = """\
You are an information-retrieval agent. You will be given:
  1. An image of a poster summarizing a source document.
  2. A JSON object called "questions" containing multiple-choice questions about that source.

Your task is to answer each question using ONLY what is visible in the poster image.
You must not use any external knowledge, prior context, or assumptions beyond what the
poster explicitly shows.

For each question:
- If the poster contains sufficient information to identify one option as correct, choose that
  option (A, B, C, or D) and cite the specific poster region that supports it (e.g., "Title
  section", "Results panel", "Bottom-right text block").
- If the poster does not contain enough information to confidently choose any option, respond
  with "NA" for both the answer and the reference. Do not guess — only mark NA when you
  genuinely cannot find supporting evidence in the poster.

Return strictly valid JSON in this exact format:
{
  "Question 1": {"answer": "A", "reference": "Top-left heading"},
  "Question 2": {"answer": "NA", "reference": "NA"},
  ...
}

Rules:
- Every question in the input must have a corresponding entry in your output.
- Do not include any text, explanations, or keys outside the JSON object.
- The "answer" field must be exactly one letter (A, B, C, or D) or the string "NA".
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchQAItem:
    qid: str
    question: str
    options: list[str]
    answer: str
    aspect: str | None
    qtype: str  # "verbatim" or "interpretive"


@dataclass
class BenchQAScore:
    raw_accuracy: float
    raw_score: int
    density_augmented: float
    correct: int
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shuffle_options(items: list[BenchQAItem], seed: int | None = None) -> list[BenchQAItem]:
    """Randomly shuffle each question's options and update the answer letter.

    LLMs systematically place the correct answer in the same position (usually
    B or C). This post-processing step guarantees a uniform A/B/C/D distribution
    regardless of what the generator produced, making the eval statistically valid.
    """
    rng = random.Random(seed)
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    shuffled: list[BenchQAItem] = []
    for item in items:
        if len(item.options) != 4:
            shuffled.append(item)
            continue

        # Find which option is currently the correct one
        correct_letter = _parse_answer_letter(item.answer)
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_idx = letter_to_idx.get(correct_letter)
        if correct_idx is None or correct_idx >= len(item.options):
            shuffled.append(item)
            continue

        correct_text = item.options[correct_idx]

        # Strip existing letter prefixes ("A. ", "B. ", etc.) then shuffle
        raw_options = [re.sub(r"^[ABCD]\.\s*", "", o) for o in item.options]
        rng.shuffle(raw_options)

        # Re-label A–D
        new_options = [f"{letter_map[i]}. {raw_options[i]}" for i in range(4)]

        # Find where the correct answer landed
        correct_raw = re.sub(r"^[ABCD]\.\s*", "", correct_text)
        new_correct_idx = raw_options.index(correct_raw)
        new_correct_letter = letter_map[new_correct_idx]
        new_answer = f"{new_correct_letter}. {correct_raw}"

        shuffled.append(BenchQAItem(
            qid=item.qid,
            question=item.question,
            options=new_options,
            answer=new_answer,
            aspect=item.aspect,
            qtype=item.qtype,
        ))
    return shuffled


def _parse_answer_letter(answer: str) -> str:
    if not answer:
        return ""
    m = re.match(r"\s*([ABCD])\b", answer.strip())
    if m:
        return m.group(1)
    return answer.strip().upper()[:1]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _truncate_for_qa(text: str, max_words: int = 8000) -> str:
    """Truncate source text to avoid exceeding LLM context limits.

    Samples the beginning (50%), end (30%), and middle (20%) so that both
    the document's opening context and its resolution/conclusion are always
    represented — critical for fiction and long-form content.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    begin_count = max_words * 5 // 10   # 50% from start
    end_count = max_words * 3 // 10     # 30% from end
    mid_count = max_words - begin_count - end_count  # 20% from middle

    begin = " ".join(words[:begin_count])

    mid_start = max(begin_count, len(words) // 2 - mid_count // 2)
    mid_end = min(len(words) - end_count, mid_start + mid_count)
    middle = " ".join(words[mid_start:mid_end])

    end = " ".join(words[-end_count:])

    return (
        begin
        + "\n\n[... document continues ...]\n\n"
        + middle
        + "\n\n[... document continues ...]\n\n"
        + end
    )


def _density_augmented_score(raw_score: float, *, length_words: int, median_words: int) -> float:
    if median_words <= 0:
        raise ValueError("median_words must be > 0")
    denom = max(1.0, length_words / float(median_words))
    return raw_score * (1.0 + 1.0 / denom)


def _validate_genre(genre: str) -> str:
    genre = genre.lower().strip()
    if genre not in GENRES:
        raise ValueError(
            f"Unknown genre {genre!r}. Must be one of: {', '.join(GENRES)}"
        )
    return genre


def _parse_qa_block(block: dict, qtype: str) -> list[BenchQAItem]:
    """Parse a {'Question N': {...}} dict into BenchQAItems."""
    items: list[BenchQAItem] = []
    if not isinstance(block, dict):
        return items

    def _sort_key(k: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", k)
        return (int(m.group(1)) if m else 10 ** 9, k)

    for key in sorted(block.keys(), key=_sort_key):
        v = block[key]
        if not isinstance(v, dict):
            continue
        items.append(BenchQAItem(
            qid=str(key),
            question=str(v.get("question", "")),
            options=[str(o) for o in (v.get("options") or [])],
            answer=str(v.get("answer", "")),
            aspect=str(v.get("aspect")) if v.get("aspect") is not None else None,
            qtype=qtype,
        ))
    return items


def _build_questions_payload(items: Iterable[BenchQAItem]) -> dict:
    payload: dict[str, dict] = {}
    for idx, item in enumerate(items, start=1):
        key = item.qid or f"Question {idx}"
        if key in payload:
            key = f"Question {idx}"
        payload[key] = {
            "aspect": item.aspect or "",
            "question": item.question,
            "options": item.options,
        }
    return payload


def _score_answers(items: list[BenchQAItem], answers: dict) -> BenchQAScore:
    correct = 0
    total = len(items)
    for idx, item in enumerate(items, start=1):
        key = item.qid if item.qid in answers else f"Question {idx}"
        ans_obj = answers.get(key, {})
        chosen    = str(ans_obj.get("answer",    "NA")).strip()
        reference = str(ans_obj.get("reference", "NA")).strip()
        # Skip if no answer given
        if chosen.upper() == "NA":
            continue
        # Skip if model answered but admitted it could not find poster evidence.
        # An answer unsupported by any cited poster region is treated as NA —
        # this prevents training-knowledge leakage from inflating scores.
        if reference.upper() == "NA":
            continue
        if _parse_answer_letter(chosen) == _parse_answer_letter(item.answer):
            correct += 1
    return BenchQAScore(
        raw_accuracy=(correct / total) if total else 0.0,
        raw_score=correct,
        density_augmented=0.0,
        correct=correct,
        total=total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QA generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_bench_qa(
    markdown: str,
    genre: str,
    *,
    provider,
) -> dict[str, list[BenchQAItem]]:
    """Generate 10 verbatim + 10 interpretive BenchQAItems from source markdown.

    Parameters
    ----------
    markdown:
        Full text of the source document (output of any any2poster parser).
    genre:
        Content genre — one of: research, news, educational, business, fiction.
    provider:
        Any provider implementing BaseLLMProvider (must have complete_json).
    """
    if not markdown or not markdown.strip():
        raise ValueError("markdown is empty — cannot generate questions")

    genre = _validate_genre(genre)
    genre_context = _GENRE_CONTEXT[genre]

    markdown = _truncate_for_qa(markdown, max_words=8000)
    # Replace straight double-quotes with typographic curly quotes so the LLM
    # never copies dialogue into JSON string values using ASCII 0x22, which
    # would produce invalid JSON (unescaped internal quotes).
    qa_source = markdown.replace('"', '“').replace('"', '”')

    verb_instructions = _VERBATIM_INSTRUCTIONS.format(genre_context=genre_context)
    interp_instructions = _INTERPRETIVE_INSTRUCTIONS.format(genre_context=genre_context)

    def _call_with_retry(system, user):
        last_exc = None
        for attempt in range(3):
            try:
                return provider.complete_json(
                    system=system,
                    user=user,
                    temperature=0.2 + attempt * 0.05,
                    max_tokens=16000,
                )
            except Exception as exc:
                last_exc = exc
        raise last_exc

    verbatim_raw = _call_with_retry(
        VERBATIM_QA_SYSTEM,
        "document_markdown:\n" + qa_source + "\n\n" + verb_instructions,
    )
    interpretive_raw = _call_with_retry(
        INTERPRETIVE_QA_SYSTEM,
        "document_markdown:\n" + qa_source + "\n\n" + interp_instructions,
    )

    verbatim_items = _shuffle_options(_parse_qa_block(verbatim_raw, "verbatim"))
    interpretive_items = _shuffle_options(_parse_qa_block(interpretive_raw, "interpretive"))

    return {
        "verbatim": verbatim_items,
        "interpretive": interpretive_items,
    }


def load_bench_qa(qa_json: str | dict) -> dict[str, list[BenchQAItem]]:
    """Load a saved bench QA JSON (written by save_bench_qa) into BenchQAItems."""
    if isinstance(qa_json, str):
        qa_obj = json.loads(qa_json)
    else:
        qa_obj = qa_json
    return {
        "verbatim": _parse_qa_block(qa_obj.get("verbatim", {}), "verbatim"),
        "interpretive": _parse_qa_block(qa_obj.get("interpretive", {}), "interpretive"),
    }


def save_bench_qa(
    qa_items: dict[str, list[BenchQAItem]],
    genre: str,
    modality: str,
    out_path: Path,
) -> None:
    """Serialize BenchQAItems to a JSON file for caching and inspection."""
    def _to_dict(items: list[BenchQAItem]) -> dict:
        return {
            item.qid: {
                "aspect": item.aspect or "",
                "question": item.question,
                "options": item.options,
                "answer": item.answer,
            }
            for item in items
        }

    payload = {
        "genre": genre,
        "modality": modality,
        "verbatim": _to_dict(qa_items["verbatim"]),
        "interpretive": _to_dict(qa_items["interpretive"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Poster answering
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_image_for_vlm(poster_image: Path, max_width: int = 3200) -> Path:
    """Return a resized copy of the poster if it exceeds max_width, else return as-is.

    GPT-4o processes images in 512px tiles. Posters rendered at print DPI
    (12000x9000px) are auto-compressed so aggressively by the API that all
    text becomes unreadable. Resizing to ~3200px wide keeps text legible while
    staying within a reasonable number of tiles.
    """
    from PIL import Image
    import tempfile

    img = Image.open(poster_image)
    w, h = img.size
    if w <= max_width:
        return poster_image

    new_w = max_width
    new_h = int(h * max_width / w)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=poster_image.parent, prefix="_eval_resized_"
    )
    tmp.close()
    resized.save(tmp.name, "PNG", optimize=False)
    return Path(tmp.name)


def answer_bench_questions(
    *,
    poster_image: Path,
    questions: dict,
    provider,
    model_name: str | None = None,
) -> dict:
    """Ask a VLM to answer MCQs using only the poster image."""
    eval_image = _prepare_image_for_vlm(poster_image)
    try:
        user = "questions:\n" + json.dumps(questions, ensure_ascii=False)
        return provider.complete_vision_json(
            system=BENCH_ANSWER_SYSTEM,
            user=user,
            image_path=eval_image,
            temperature=0.0,
            max_tokens=4096,
            model_override=model_name,
        )
    finally:
        if eval_image != poster_image and eval_image.exists():
            eval_image.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation orchestration
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bench_quiz(
    *,
    poster_image: Path,
    poster_text: str,
    qa_items: dict[str, list[BenchQAItem]],
    genre: str,
    modality: str,
    provider,
    model_names: Iterable[str],
    median_words: int = 774,
) -> dict:
    """Run BenchQuiz evaluation across one or more VLM readers.

    Parameters
    ----------
    poster_image:
        PNG of the generated poster.
    poster_text:
        Extracted plain text of the poster (for density-augmented score).
    qa_items:
        Output of generate_bench_qa or load_bench_qa.
    genre:
        Content genre label (stored in results for downstream analysis).
    modality:
        Input modality label (stored in results for downstream analysis).
    provider:
        Provider implementing BaseLLMProvider + complete_vision_json.
    model_names:
        VLM model identifiers used as readers; results are averaged across models.
    median_words:
        Median poster word count for density normalisation. Default 774 matches
        the Paper2Poster baseline so scores are directly comparable on research genre.
    """
    length_words = _word_count(poster_text)
    model_names = list(model_names)

    results: dict = {
        "genre": genre,
        "modality": modality,
        "by_model": {},
        "avg": {},
    }

    for model in model_names:
        verb_payload = _build_questions_payload(qa_items["verbatim"])
        interp_payload = _build_questions_payload(qa_items["interpretive"])

        verb_answers = answer_bench_questions(
            poster_image=poster_image,
            questions=verb_payload,
            provider=provider,
            model_name=model,
        )
        interp_answers = answer_bench_questions(
            poster_image=poster_image,
            questions=interp_payload,
            provider=provider,
            model_name=model,
        )

        verb_score = _score_answers(qa_items["verbatim"], verb_answers)
        interp_score = _score_answers(qa_items["interpretive"], interp_answers)

        verb_score.density_augmented = _density_augmented_score(
            verb_score.raw_score, length_words=length_words, median_words=median_words
        )
        interp_score.density_augmented = _density_augmented_score(
            interp_score.raw_score, length_words=length_words, median_words=median_words
        )

        overall_correct = verb_score.correct + interp_score.correct
        overall_total = max(1, verb_score.total + interp_score.total)
        overall_accuracy = overall_correct / overall_total
        overall_aug = _density_augmented_score(
            overall_correct, length_words=length_words, median_words=median_words
        )

        results["by_model"][model] = {
            "verbatim": verb_score.__dict__,
            "interpretive": interp_score.__dict__,
            "overall_correct": overall_correct,
            "overall_accuracy": overall_accuracy,
            "overall_aug": overall_aug,
            "answers": {
                "verbatim": verb_answers,
                "interpretive": interp_answers,
            },
        }

    if results["by_model"]:
        verb_raws = [v["verbatim"]["raw_accuracy"] for v in results["by_model"].values()]
        interp_raws = [v["interpretive"]["raw_accuracy"] for v in results["by_model"].values()]
        overall_accs = [v["overall_accuracy"] for v in results["by_model"].values()]
        overall_augs = [v["overall_aug"] for v in results["by_model"].values()]
        results["avg"] = {
            "verbatim_raw": sum(verb_raws) / len(verb_raws),
            "interpretive_raw": sum(interp_raws) / len(interp_raws),
            "overall_accuracy": sum(overall_accs) / len(overall_accs),
            "overall_aug": sum(overall_augs) / len(overall_augs),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Poster text extraction
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_TITLE_RE = re.compile(
    r'class="banner-title"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL
)
_BANNER_AUTHORS_RE = re.compile(
    r'class="banner-authors"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL
)


def extract_poster_text(html_text: str) -> str:
    """Extract plain text from a poster HTML for word-count / density scoring."""
    if not html_text:
        return ""

    from any2poster.eval.stats import extract_panels_from_html

    parts: list[str] = []
    m = _BANNER_TITLE_RE.search(html_text)
    if m:
        parts.append(m.group(1).strip())
    m = _BANNER_AUTHORS_RE.search(html_text)
    if m:
        parts.append(m.group(1).strip())
    for panel in extract_panels_from_html(html_text):
        if panel.text:
            parts.append(panel.text)
    return "\n".join(p for p in parts if p).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "any2poster BenchQuiz — domain-neutral QA evaluation.\n"
            "Generates 10+10 MCQs from source text, answers from poster only."
        )
    )
    ap.add_argument("--source-text", required=True,
                    help="Path to source document text (output of any2poster parser)")
    ap.add_argument("--poster-image", required=True,
                    help="Path to poster PNG")
    ap.add_argument("--poster-html", required=True,
                    help="Path to poster HTML (for density-augmented score)")
    ap.add_argument("--genre", required=True, choices=list(GENRES),
                    help="Content genre of the source document")
    ap.add_argument("--modality", required=True,
                    help="Input modality label (e.g. pdf, url, pptx, video)")
    ap.add_argument("--models", default="openai/gpt-4o",
                    help="Comma-separated VLM model names for answering")
    ap.add_argument("--qa-json", default="",
                    help="Path to existing bench QA JSON (skips generation)")
    ap.add_argument("--qa-out", default="",
                    help="Path to save generated QA JSON for caching")
    ap.add_argument("--out", default="",
                    help="Path to save evaluation results JSON")
    ap.add_argument("--median-words", type=int, default=774,
                    help="Median poster word count for density-augmented score (default: 774)")
    args = ap.parse_args()

    from any2poster.providers.openrouter import OpenRouterProvider
    provider = OpenRouterProvider()

    source_text = Path(args.source_text).read_text(encoding="utf-8")
    poster_html_text = Path(args.poster_html).read_text(encoding="utf-8")
    poster_text = extract_poster_text(poster_html_text)

    if args.qa_json and Path(args.qa_json).exists():
        print(f"[bench_quiz] Loading QA from: {args.qa_json}")
        qa_items = load_bench_qa(Path(args.qa_json).read_text(encoding="utf-8"))
    else:
        print(f"[bench_quiz] Generating QA  genre={args.genre!r} modality={args.modality!r} …")
        qa_items = generate_bench_qa(source_text, args.genre, provider=provider)
        verb_n = len(qa_items["verbatim"])
        interp_n = len(qa_items["interpretive"])
        print(f"[bench_quiz] Generated {verb_n} verbatim + {interp_n} interpretive questions")
        if args.qa_out:
            save_bench_qa(qa_items, args.genre, args.modality, Path(args.qa_out))
            print(f"[bench_quiz] QA saved: {args.qa_out}")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[bench_quiz] Evaluating with: {model_list}")

    results = evaluate_bench_quiz(
        poster_image=Path(args.poster_image),
        poster_text=poster_text,
        qa_items=qa_items,
        genre=args.genre,
        modality=args.modality,
        provider=provider,
        model_names=model_list,
        median_words=args.median_words,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench_quiz] Results saved: {args.out}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    avg = results.get("avg", {})
    print()
    print("── BenchQuiz Summary " + "─" * 40)
    print(f"  Genre              : {args.genre}")
    print(f"  Modality           : {args.modality}")
    print(f"  Verbatim accuracy  : {avg.get('verbatim_raw', 0.0):.1%}")
    print(f"  Interpretive acc.  : {avg.get('interpretive_raw', 0.0):.1%}")
    print(f"  Overall accuracy   : {avg.get('overall_accuracy', 0.0):.1%}")
    print(f"  Overall (aug.)     : {avg.get('overall_aug', 0.0):.4f}")
    print("─" * 60)


if __name__ == "__main__":
    _cli()

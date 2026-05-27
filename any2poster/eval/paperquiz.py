"""PaperQuiz evaluation (information extraction) - Paper2Poster-aligned.

Implements:
  1) QA reuse from Paper2Poster dataset (preferred for comparability)
  2) Optional QA generation from paper markdown using the exact prompts
  3) Poster-only answering with strict JSON + NA rules
  4) Raw accuracy + density-augmented score
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable


# ----------------------------
# Paper2Poster prompts (Appendix F.4)
# ----------------------------


VERBATIM_QA_SYSTEM = (
    "You are a Question-Generation agent for academic posters. Your task is to read "
    "the supplied Markdown text (document_markdown) and produce exactly 50 multiple-choice "
    "QA items whose answers can be located verbatim or nearly verbatim in that text. The "
    "questions must be suitable for conference-poster readers: avoid deep theoretical proofs, "
    "reference lists, or citation minutiae. Follow all guidelines below precisely."
)

VERBATIM_QA_INSTRUCTIONS = """Instructions:
1. Carefully read the Markdown in document_markdown.
- Each question must map to one clear sentence or phrase in the poster text.
- No duplicate or near-duplicate wording.
2. Write 50 factual, answerable-from-text questions.
- Vary difficulty from easy "headline" facts to specific numeric or procedural details.
3. Distribute the 50 questions across the following poster-friendly aspects, aiming for 2-5 questions per aspect and ensuring each aspect appears at least once:
A. Title & authorship (title, author names, affiliations, keywords)
B. Motivation / problem statement / research gap
C. Objectives or hypotheses
D. Dataset(s) or experimental materials
E. Methodology (algorithms, model architecture, workflow steps)
F. Key parameters or hyper-parameters (values, settings)
G. Evaluation metrics or criteria
H. Quantitative results (numbers in tables, charts)
I. Qualitative findings, figures, or illustrative examples
J. Comparative or ablation study results
K. Conclusions, implications, or contributions
L. Limitations or future work
M. Definitions of domain-specific terms or abbreviations
4. EXCLUDE references, citations, author acknowledgements, and any text that would not appear on a standard poster.
5. Use the following JSON-for-each format (exact spelling & casing):
{
"Question X": {
"aspect": "<A-M>",
"question": "<single sentence>",
"options": [
"A. <choice 1>",
"B. <choice 2>",
"C. <choice 3>",
"D. <choice 4>"
],
"answer": "<Letter>. <exact correct option text>"
},
...
}
6. Output only the final JSON object containing 50 items-no additional commentary.
7. Balance the correct answers roughly equally among options A-D.
Think step by step and ensure full compliance with every guideline.
"""


INTERPRETIVE_QA_SYSTEM = (
    "You are a Question-Generation agent. Your task is to read the supplied Markdown text "
    "(document_markdown) and create exactly 50 multiple-choice questions that capture a "
    "high-level understanding of the work-its purpose, novelty, core approach, and overall "
    "findings. Every question must still be answerable by locating explicit sentences or "
    "phrases in the text; do not require inference that is absent from the poster-style content."
)

INTERPRETIVE_QA_INSTRUCTIONS = """Instructions:
1. Read the Markdown in document_markdown closely.
- Each question must map to explicit content in the text.
- Do not require inference beyond presented poster-level information.
2. Draft 50 factual questions probing the reader's global grasp (e.g., "What problem does the study address"").
- Avoid low-level numeric settings, code snippets, or reference lists.
- Vary wording and avoid duplicates.
3. Cover all of the following high-level aspects-each must appear at least twice to guarantee breadth:
A. Research domain & background context
B. Central problem / motivation / research gap
C. Primary goal, hypothesis, or research question
D. Key contributions or novelty statements
E. Overall methodology or workflow (summarized)
F. Principal findings or headline quantitative results
G. Qualitative insights or illustrative examples
H. Implications, applications, or significance
I. Limitations or future-work directions
J. Main conclusions or take-home messages
4. EXCLUDE citations, granular hyper-parameters, precise numeric tables, and acknowledgements-stick to poster-level overview content.
5. Return the questions in the following strict JSON schema:
{
"Question X": {
"aspect": "<A-J>",
"question": "<one concise sentence>",
"options": [
"A. <choice 1>",
"B. <choice 2>",
"C. <choice 3>",
"D. <choice 4>"
],
"answer": "<Letter>. <exact correct option text>"
},
...
}
6. Produce only the final JSON object with 50 entries-no commentary, headers, or extra lines.
7. The number of correct answers should be approximately balanced across A-D.
"""


ANSWER_SYSTEM = """You are an answering agent. You will be provided with:
1. An image of a poster.
2. A JSON object called "questions" which contains multiple questions. Each question has four
possible answers: A, B, C, or D.
Your goal is to analyze the poster thoroughly and answer each question based on the information
it provides. You should NOT use any external knowledge or context beyond the poster image.
You must rely solely on the content of the poster to answer the questions.
For each question:
- If you find enough evidence in the poster to decide on a specific option (A, B, C, or D), then
  choose that option and include a brief reference to the part of the poster that supports your answer
  (e.g., "Top-left text", "Event date section", etc.).
- If the poster does not offer sufficient information to confidently choose any of the options,
  respond with "NA" for both the answer and the reference.
Format your output strictly as a JSON object with this pattern:
{
"Question 1": {"answer": "X", "reference": "some reference or 'NA'"},
"Question 2": {"answer": "X", "reference": "some reference or 'NA'"},
...
}
Do not include any explanations or extra keys beyond the specified structure.
You must provide an answer entry for all questions in the "questions" object.
"""


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class QAItem:
    qid: str
    question: str
    options: list[str]
    answer: str
    aspect: str | None
    qtype: str  # "verbatim" or "interpretive"


@dataclass
class QAScore:
    raw_accuracy: float
    raw_score: int
    density_augmented: float
    correct: int
    total: int


# ----------------------------
# QA parsing
# ----------------------------


def _parse_answer_letter(answer: str) -> str:
    if not answer:
        return ""
    m = re.match(r"\s*([ABCD])\b", answer.strip())
    if m:
        return m.group(1)
    return answer.strip().upper()[:1]


def _normalize_qa_block(block: dict, qtype: str) -> list[QAItem]:
    items: list[QAItem] = []
    if not block:
        return items

    # Paper2Poster format used by eval code: {"questions": [...], "answers": [...], "aspects": [...]}
    if isinstance(block, dict) and "questions" in block:
        questions = block.get("questions", {})
        answers = block.get("answers", {})
        aspects = block.get("aspects", {})

        def _sorted_keys(qdict: dict) -> list[str]:
            keys = list(qdict.keys())
            def _key_fn(k: str) -> tuple[int, str]:
                m = re.search(r"(\\d+)", k)
                return (int(m.group(1)) if m else 10**9, k)
            return sorted(keys, key=_key_fn)

        # Case 1: dict-of-questions keyed by "Question X"
        if isinstance(questions, dict):
            for k in _sorted_keys(questions):
                q = questions.get(k, {})
                opts = q.get("options") if isinstance(q, dict) else None
                if opts is None and isinstance(q, dict):
                    opts = q.get("choices")
                opts = opts or []
                if isinstance(answers, dict):
                    ans = answers.get(k, q.get("answer", ""))
                else:
                    ans = ""
                if isinstance(aspects, dict):
                    aspect = aspects.get(k, q.get("aspect"))
                else:
                    aspect = None
                items.append(
                    QAItem(
                        qid=str(k),
                        question=q.get("question", "") if isinstance(q, dict) else str(q),
                        options=[str(o) for o in opts],
                        answer=str(ans),
                        aspect=str(aspect) if aspect is not None else None,
                        qtype=qtype,
                    )
                )
            return items

        # Case 2: list-of-questions
        if isinstance(questions, list):
            for i, q in enumerate(questions):
                qid = f"Question {i+1}"
                opts = q.get("options") if isinstance(q, dict) else None
                if opts is None and isinstance(q, dict):
                    opts = q.get("choices")
                opts = opts or []
                if isinstance(answers, dict):
                    ans = answers.get(qid, q.get("answer", ""))
                else:
                    ans = answers[i] if i < len(answers) else q.get("answer", "")
                if isinstance(aspects, dict):
                    aspect = aspects.get(qid, q.get("aspect"))
                else:
                    aspect = aspects[i] if i < len(aspects) else q.get("aspect")
                items.append(
                    QAItem(
                        qid=qid,
                        question=q.get("question", "") if isinstance(q, dict) else str(q),
                        options=[str(o) for o in opts],
                        answer=str(ans),
                        aspect=str(aspect) if aspect is not None else None,
                        qtype=qtype,
                    )
                )
            return items

    # Prompt-style format: {"Question 1": {...}, "Question 2": {...}}
    if isinstance(block, dict):
        for k, v in block.items():
            if not isinstance(v, dict):
                continue
            items.append(
                QAItem(
                    qid=str(k),
                    question=str(v.get("question", "")),
                    options=[str(o) for o in (v.get("options") or [])],
                    answer=str(v.get("answer", "")),
                    aspect=str(v.get("aspect")) if v.get("aspect") is not None else None,
                    qtype=qtype,
                )
            )
    return items


def load_paper2poster_qa(qa_json: str | dict) -> dict[str, list[QAItem]]:
    """Parse Paper2Poster QA JSON into verbatim + interpretive lists."""
    if isinstance(qa_json, str):
        qa_obj = json.loads(qa_json)
    else:
        qa_obj = qa_json

    # Paper2Poster uses "detail" and "understanding"
    detail = qa_obj.get("detail", qa_obj.get("verbatim", {}))
    understanding = qa_obj.get("understanding", qa_obj.get("interpretive", {}))

    verbatim_items = _normalize_qa_block(detail, "verbatim")
    interpretive_items = _normalize_qa_block(understanding, "interpretive")
    return {"verbatim": verbatim_items, "interpretive": interpretive_items}


def build_questions_payload(items: Iterable[QAItem]) -> dict:
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


# ----------------------------
# Poster length (for density-augmented score)
# ----------------------------


def _word_count(text: str) -> int:
    """Approximate word count for density normalization (Paper2Poster uses words)."""
    text = text or ""
    return len(re.findall(r"\b\w+\b", text))


def density_augmented_score(raw_score: float, *, length_words: int, median_words: int) -> float:
    if median_words <= 0:
        raise ValueError("median_words must be > 0")
    denom = max(1.0, length_words / float(median_words))
    return raw_score * (1.0 + 1.0 / denom)


# ----------------------------
# QA generation (optional, for non-dataset papers)
# ----------------------------


def generate_qa_from_markdown(markdown: str, *, provider) -> dict[str, list[QAItem]]:
    """Generate verbatim + interpretive questions using the Paper2Poster prompts."""
    if not markdown:
        raise ValueError("markdown is empty")

    verb_user = "document_markdown:\n" + markdown + "\n\n" + VERBATIM_QA_INSTRUCTIONS
    verbatim = provider.complete_json(
        system=VERBATIM_QA_SYSTEM,
        user=verb_user,
        temperature=0.2,
        max_tokens=8192,
    )

    interp_user = "document_markdown:\n" + markdown + "\n\n" + INTERPRETIVE_QA_INSTRUCTIONS
    interpretive = provider.complete_json(
        system=INTERPRETIVE_QA_SYSTEM,
        user=interp_user,
        temperature=0.2,
        max_tokens=8192,
    )

    return {
        "verbatim": _normalize_qa_block(verbatim, "verbatim"),
        "interpretive": _normalize_qa_block(interpretive, "interpretive"),
    }


# ----------------------------
# Answering + scoring
# ----------------------------


def answer_questions_with_vlm(
    *,
    poster_image: Path,
    questions: dict,
    provider,
    model_name: str | None = None,
) -> dict:
    """Ask a VLM to answer questions using only the poster image."""
    user = "questions:\n" + json.dumps(questions, ensure_ascii=False)
    return provider.complete_vision_json(
        system=ANSWER_SYSTEM,
        user=user,
        image_path=poster_image,
        temperature=0.0,
        max_tokens=4096,
        model_override=model_name,
    )


def _score_answers(items: list[QAItem], answers: dict) -> QAScore:
    correct = 0
    total = len(items)
    for idx, item in enumerate(items, start=1):
        key = item.qid if item.qid in answers else f"Question {idx}"
        ans_obj = answers.get(key, {})
        chosen = str(ans_obj.get("answer", "NA")).strip()
        if chosen.upper() == "NA":
            continue
        chosen_letter = _parse_answer_letter(chosen)
        gold_letter = _parse_answer_letter(item.answer)
        if chosen_letter == gold_letter:
            correct += 1
    raw_acc = (correct / total) if total else 0.0
    return QAScore(
        raw_accuracy=raw_acc,
        raw_score=correct,
        density_augmented=0.0,
        correct=correct,
        total=total,
    )


def evaluate_paperquiz(
    *,
    poster_image: Path,
    poster_text: str,
    qa_items: dict[str, list[QAItem]],
    provider,
    model_names: Iterable[str],
    median_words: int = 774,
) -> dict:
    """Evaluate PaperQuiz with multiple readers and return aggregated metrics."""
    length_words = _word_count(poster_text)

    results = {"by_model": {}, "avg": {}}
    for model in model_names:
        verb_payload = build_questions_payload(qa_items["verbatim"])
        interp_payload = build_questions_payload(qa_items["interpretive"])

        verb_answers = answer_questions_with_vlm(
            poster_image=poster_image, questions=verb_payload, provider=provider, model_name=model
        )
        interp_answers = answer_questions_with_vlm(
            poster_image=poster_image, questions=interp_payload, provider=provider, model_name=model
        )

        verb_score = _score_answers(qa_items["verbatim"], verb_answers)
        interp_score = _score_answers(qa_items["interpretive"], interp_answers)

        verb_score.density_augmented = density_augmented_score(
            verb_score.raw_score, length_words=length_words, median_words=median_words
        )
        interp_score.density_augmented = density_augmented_score(
            interp_score.raw_score, length_words=length_words, median_words=median_words
        )
        overall_score = verb_score.correct + interp_score.correct
        overall_total = max(1, verb_score.total + interp_score.total)
        overall_accuracy = overall_score / overall_total
        overall_aug = density_augmented_score(
            overall_score, length_words=length_words, median_words=median_words
        )

        results["by_model"][model] = {
            "verbatim": verb_score.__dict__,
            "interpretive": interp_score.__dict__,
            "overall_score": overall_score,
            "overall_accuracy": overall_accuracy,
            "overall_aug": overall_aug,
            "answers": {
                "verbatim": verb_answers,
                "interpretive": interp_answers,
            },
        }

    # Aggregate across models
    if results["by_model"]:
        verb_raw = [v["verbatim"]["raw_accuracy"] for v in results["by_model"].values()]
        interp_raw = [v["interpretive"]["raw_accuracy"] for v in results["by_model"].values()]
        overall_scores = [v["overall_score"] for v in results["by_model"].values()]
        overall_accs = [v["overall_accuracy"] for v in results["by_model"].values()]
        overall_augs = [v["overall_aug"] for v in results["by_model"].values()]
        results["avg"] = {
            "verbatim_raw": sum(verb_raw) / len(verb_raw),
            "interpretive_raw": sum(interp_raw) / len(interp_raw),
            "overall_score": sum(overall_scores) / len(overall_scores),
            "overall_accuracy": sum(overall_accs) / len(overall_accs),
            "overall_aug": sum(overall_augs) / len(overall_augs),
        }

    return results


# ----------------------------
# CLI
# ----------------------------


def _cli() -> None:
    import argparse
    from any2poster.providers.openrouter import OpenRouterProvider
    parser = argparse.ArgumentParser(description="any2poster PaperQuiz evaluator (Paper2Poster-style).")
    parser.add_argument("--poster-image", required=True)
    parser.add_argument("--poster-html", required=True)
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--median-words", type=int, default=774)
    parser.add_argument("--median-tokens", dest="median_words", type=int, help="Deprecated alias for --median-words")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    poster_html = Path(args.poster_html).read_text(encoding="utf-8")
    poster_text = extract_poster_text_from_html(poster_html)

    qa_obj = json.loads(Path(args.qa_json).read_text(encoding="utf-8"))
    qa_items = load_paper2poster_qa(qa_obj)

    provider = OpenRouterProvider()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    results = evaluate_paperquiz(
        poster_image=Path(args.poster_image),
        poster_text=poster_text,
        qa_items=qa_items,
        provider=provider,
        model_names=model_list,
        median_words=args.median_words,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2))


# ----------------------------
# Poster text extraction
# ----------------------------


_BANNER_TITLE_RE = re.compile(r'class="banner-title"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL)
_BANNER_AUTHORS_RE = re.compile(r'class="banner-authors"\s*>\s*(.*?)\s*<', re.IGNORECASE | re.DOTALL)


def extract_poster_text_from_html(html_text: str) -> str:
    """Extract banner title/authors + panel text for length calculation."""
    if not html_text:
        return ""

    from any2poster.eval.stats import extract_panels_from_html  # local import to avoid cycles

    parts: list[str] = []
    title_match = _BANNER_TITLE_RE.search(html_text)
    if title_match:
        parts.append(title_match.group(1).strip())
    authors_match = _BANNER_AUTHORS_RE.search(html_text)
    if authors_match:
        parts.append(authors_match.group(1).strip())

    panels = extract_panels_from_html(html_text)
    for p in panels:
        if p.text:
            parts.append(p.text)

    return "\n".join(p for p in parts if p).strip()


if __name__ == "__main__":
    _cli()

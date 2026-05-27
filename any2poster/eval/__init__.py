"""Evaluation utilities for any2poster.

Phase 1 (stats) mirrors Paper2Poster "stats" metrics:
- CLIP_similarity (AltCLIP cosine similarity between GT and generated posters)
- textual_ppl (Llama-2-7b-hf)
- mixtual_ppl (PPL on markdown with image refs)
- visual_relevance (AltCLIP image↔text similarity for panel visuals)
- visual_ppl / interleaved_ppl / poster_image_ppl (Qwen2.5-VL vLLM-based)

Phase 2 (PaperQuiz) mirrors Paper2Poster Appendix F.4:
- 50 verbatim + 50 interpretive MCQs (o3 generation or dataset reuse)
- Poster-only VLM answering with NA rule
- Raw accuracy + density-augmented score
"""

from __future__ import annotations

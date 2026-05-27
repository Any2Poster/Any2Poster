"""Pipeline stages for any2poster.

Each stage is a separate module with a run_*_stage entry point:
  parse_stage    — Extract content from documents
  analyze_stage  — Two-pass LLM analysis
  plan_stage     — 3-column grid layout
  generate_stage — Panel image generation via Nano Banana Pro
  compile_stage  — Assemble final PDF/PNG
"""

from any2poster.stages.analyze_stage import run_analyze_stage
from any2poster.stages.compile_stage import run_compile_stage
from any2poster.stages.generate_stage import run_generate_stage
from any2poster.stages.plan_stage import run_plan_stage

__all__ = [
    "run_analyze_stage",
    "run_compile_stage",
    "run_generate_stage",
    "run_plan_stage",
]
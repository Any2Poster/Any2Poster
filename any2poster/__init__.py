"""any2poster — Universal document to poster pipeline.

Convert any input (PDF, DOCX, Markdown, HTML, URL) into a
conference-quality academic poster using AI-powered analysis
and Nano Banana Pro image generation.
"""

__version__ = "0.2.0"

from any2poster.models import PosterConfig
from any2poster.pipeline import run_pipeline

__all__ = ["PosterConfig", "run_pipeline", "__version__"]
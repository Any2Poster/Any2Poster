# 🎓 Any2Poster: Any-Source Poster Generation Across Modalities and Domains

> *Turn any document, slideshow, notebook, or video into a publication-quality academic poster.*

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-red)](https://arxiv.org)
[![Project Page](https://img.shields.io/badge/Project-Page-brightgreen)](https://any2poster.github.io)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Any2Poster/Any2Poster-Bench)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

We address **How to generate a poster from any input source** and **How to evaluate poster quality across domains**.

---

## 🤩 Any2Poster Generating Its Own Poster

[![Any2Poster Example](assets/aaai_poster6.png)](assets/aaai_poster6.png)

---

## ✨ Key Contributions

- 📄 **Universal Input Abstraction** — A single normalized schema ingests PDF, DOCX, PPTX, LaTeX, Markdown, Jupyter Notebook, HTML/URL, and YouTube video. No source-specific pipelines.
- 🖥️ **Deterministic HTML/CSS Rendering** — All text is rendered pixel-accurately via Playwright. Unlike prior work that pipes text through image generation models, Any2Poster guarantees zero typography hallucination.
- 🔁 **Panel-Level VLM Feedback Loop** — Playwright renders each panel to PNG → a VLM inspects it → structured error codes from a fixed vocabulary (overflow, blank space, font size, figure balance, reading order) → deterministic CSS mutations applied → only failing panels re-rendered. Closes the **engagement bottleneck** explicitly identified as unsolved in Paper2Poster.
- 🧠 **Data Extractor Validation** — Chart values are grounded verbatim in source text before image generation, eliminating fabricated statistics.
- 🖼️ **Hybrid Figure Strategy** — Original extracted figures are preserved and placed; AI generation is reserved only for panels with no suitable extracted figure.
- 📊 **Any2Poster Bench** — A multi-format, cross-domain benchmark with genre-aware MCQ generation, BenchQuiz scoring, and VLM-as-diverse-readers evaluation across 6 criteria.

---

## 🔥 Updates

- **[2025.05.27]** Initial public release of Any2Poster v0.3.0 — code, benchmark, and eval pipeline.

---

## 🏗️ Pipeline Overview

Any2Poster transforms any input through a **6-stage checkpointed pipeline**:

| Stage | Name | Description | Checkpoint |
|-------|------|-------------|------------|
| 1 | 🔍 **Parse** | Extract text, figures, tables from any input format via Docling | `checkpoint_parse.json` |
| 2 | ✂️ **Chunk** | Segment into semantically coherent sections with type classification | `checkpoint_chunk.json` |
| 3 | 🧠 **Analyze** | Two-pass LLM analysis — global poster strategy then per-section bullet extraction | `checkpoint_analyze.json` |
| 4 | 📐 **Plan** | 3-column grid layout with panel category assignment and figure allocation | `checkpoint_plan.json` |
| 5 | 🎨 **Generate** | Panel-level HTML/CSS generation + AI visual synthesis via Gemini 3 Pro | `checkpoint_generate.json` |
| 6 | 📄 **Compile** | Playwright renders final print-quality PDF and PNG with VLM feedback loop | Output |

> 💾 Every stage is checkpointed automatically. Run the same command again to resume from where you left off.

---

## ⚡ Comparison with Prior Work

| Feature | Paper2Poster | Paper2Slides | **Any2Poster** |
|---------|-------------|--------------|----------------|
| Input formats | PDF only | PDF, DOCX, PPTX, MD | **PDF, DOCX, PPTX, LaTeX, MD, ipynb, HTML, Video** |
| Output format | PPTX | PDF (image-rendered) | **PDF + PNG + HTML artifact** |
| Text rendering | PPTX XML | AI image generation | **Deterministic HTML/CSS via Playwright** |
| VLM feedback loop | ✅ Panel-level | ❌ | **✅ Panel-level with fixed error vocabulary** |
| Evaluation benchmark | PaperQuiz (PDF only) | ❌ None | **Any2Poster Bench (multi-format, cross-domain)** |
| Checkpoint / resume | ❌ | ✅ | **✅ Full 6-stage checkpointing** |
| Web UI | ❌ | ✅ | 🔜 Coming soon |

---

## 🛠️ Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/Any2Poster/Any2Poster.git
cd Any2Poster

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[render]"

# Install Playwright Chromium browser (required for rendering)
playwright install chromium
```

**API Keys**

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

The only required key to get started is `OPENROUTER_API_KEY`. Get one at [openrouter.ai](https://openrouter.ai).

---

## 🚀 Quick Start

**Simple run — any input to poster:**
```bash
any2poster paper.pdf
```

**Specify output path:**
```bash
any2poster paper.pdf -o results/my_poster.pdf
```

**Use a specific color theme:**
```bash
any2poster paper.pdf --style arctic -o poster.pdf
```

**Enable the VLM feedback loop:**
```bash
any2poster paper.pdf --feedback --feedback-iters 2 --dpi 250 -o poster.pdf
```

**Resume from cached checkpoints:**
```bash
any2poster paper.pdf --resume -o poster.pdf
```

**Generate AI diagrams for panels with no extracted figure:**
```bash
any2poster paper.pdf --generate-visuals --feedback -o poster.pdf
```

---

## 📄 Supported Inputs

Any2Poster accepts any of the following:

| Format | Example |
|--------|---------|
| 📄 PDF | `paper.pdf` |
| 📝 Word Document | `report.docx` |
| 📊 PowerPoint | `slides.pptx` |
| 🧪 LaTeX | `manuscript.tex` |
| 📋 Markdown / Plain Text | `notes.md`, `notes.txt` |
| 🔬 Jupyter Notebook | `experiments.ipynb` |
| 🌐 HTML or URL | `index.html`, `https://arxiv.org/abs/...` |
| 🎥 Video (YouTube URL) | `https://youtube.com/watch?v=...` |

---

## 🎨 Color Themes

| Theme | Description |
|-------|-------------|
| `light` | Clean white background — default |
| `arctic` | White and blue academic style |
| `academic_gray` | Traditional conference poster grey |
| `steel_blue` | Deep blue professional |
| `periwinkle` | Soft blue-purple |
| `forest` | Deep green |
| `onyx` | Dark charcoal |
| `aquamarine` | Teal accent |
| `ruby` | Deep red |
| `dark` | Dark mode (randomly selects from frozen_lake, neon_ice, bubblegum_dark) |
| `auto` | Infers best theme from document content |
| `custom` | Full manual hex control (see below) |

**Custom colors:**
```bash
any2poster paper.pdf \
  --style custom \
  --bg-color "#F5F5F5" \
  --header-color "#1A237E" \
  --subheader-color "#3949AB" \
  --header-text-color "#FFFFFF" \
  --text-color "#1A1A1A"
```

**Add logos to your poster:**
```bash
any2poster paper.pdf \
  --logo-left path/to/conference_logo.png \
  --logo-right path/to/institution_logo.png
```

---

## ⚙️ Full CLI Reference
Usage: any2poster [OPTIONS] INPUT_PATH
Options:
-o, --output PATH                     Output file (.pdf or .png)
--style STYLE                         Color theme [default: light]
--mode [professional|casual]          Layout mode [default: professional]
--width FLOAT                         Poster width in inches [default: 48.0]
--height FLOAT                        Poster height in inches [default: 36.0]
--dpi INT                             Output resolution [default: 150]
--logo-left PATH                      Left banner logo
--logo-right PATH                     Right banner logo
--resume / --no-resume                Resume from checkpoint [default: no-resume]
--debug / --no-debug                  Verbose output
--generate-visuals                    Generate AI diagrams via Gemini 3 Pro
--llm-model TEXT                      LLM model ID override
--image-model TEXT                    Image model ID override
--vision-model TEXT                   Vision model for quality checks
--feedback / --no-feedback            Enable panel VLM feedback loop
--feedback-iters INT                  Max feedback iterations [default: 2]
--feedback-model TEXT                 Vision model for feedback loop
--quality-check / --no-quality-check  Validate generated visuals
--quality-retries INT                 Extra retries when quality fails
--bg-color TEXT                       Custom background hex
--header-color TEXT                   Custom panel header hex
--header-text-color TEXT              Custom header text hex
--text-color TEXT                     Custom body text hex
--interactive                         Prompt for style and mode interactively

---

## 🔮 Evaluation

Any2Poster ships with a full evaluation suite mirroring and extending Paper2Poster's benchmark methodology.

### Run full dataset evaluation

```bash
python any2poster-eval/run_p2p_first_eval.py \
  --index 0 \
  --run-all \
  --feedback \
  --feedback-iters 2 \
  --disable-vlm-ppl
```

### BenchQuiz (information fidelity)

```bash
python -m any2poster.eval.paperquiz \
  --poster-image path/to/poster.png \
  --poster-html path/to/poster.html \
  --qa-json path/to/qa.json \
  --models "openai/gpt-4o" \
  --out eval_results/paper_id/any2poster/benchquiz.json
```

### VLM-as-Judge (holistic quality)

```bash
python -m any2poster.eval.vlm_judge \
  --poster-image path/to/poster.png \
  --model "openai/gpt-4o" \
  --out eval_results/paper_id/any2poster/vlm_judge.json
```

### Statistical metrics (AltCLIP + Llama perplexity)

```bash
python -m any2poster.eval.stats \
  --paper-id paper_id \
  --method-name any2poster \
  --gen-poster-png path/to/poster.png \
  --gen-poster-html path/to/poster.html \
  --gt-poster-png path/to/gt.png \
  --disable-vlm-ppl
```

### Generate BenchQuiz questions for your own paper

```bash
python -m any2poster.eval.paperquiz \
  --paper-folder data/your_paper_name \
  --generate-questions
```

---

## 📊 Any2Poster Bench

Any2Poster Bench is a **multi-format, cross-domain** benchmark for academic poster generation, extending Paper2Poster's PDF-only PaperQuiz with:

- 🎯 **Genre-aware MCQ generation** — question difficulty and style adapts to content domain
- 📝 **BenchQuiz scoring** — verbatim and interpretive accuracy with density-augmented scoring
- 👁️ **VLM-as-diverse-readers** — multiple VLM reader personas simulate varied audience expertise
- ⚖️ **6-criteria VLM-as-Judge** — layout, readability, figure relevance, content fidelity, visual coherence, engagement
- 🖼️ **AltCLIP figure relevance** — measures whether each figure is placed next to semantically matching text
- 📖 **Llama-2-7b perplexity** — measures textual coherence of the poster's text flow

The full dataset is available on HuggingFace: 🤗 [Any2Poster/Any2Poster-Bench](https://huggingface.co/datasets/Any2Poster/Any2Poster-Bench)

---

## 🐳 Docker

```bash
# Build
docker build -t any2poster .

# Run
docker run --rm \
  -e OPENROUTER_API_KEY=your_key_here \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  any2poster /input/paper.pdf -o /output/poster.pdf --feedback
```

---

## ❤️ Acknowledgements

We thank [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster), [Paper2Slides](https://github.com/HKUDS/Paper2Slides), [Docling](https://github.com/docling-project/docling), and [Playwright](https://playwright.dev/) for their outstanding open-source contributions which this work builds upon.

---

## 📖 Citation

If you find Any2Poster useful in your research, please cite:

```bibtex
@misc{any2poster2025,
  title     = {Any2Poster: Any-Source Poster Generation Across Modalities and Domains},
  author    = {Vinaykumar, Amogh and Li, Aiden and Huang, Suozhi and Liu, Shilong},
  year      = {2025},
  url       = {https://github.com/Any2Poster/Any2Poster}
}
```

---

<p align="center">
  Made by Amogh Vinaykumar, Aiden Li, Suozhi Huang, and Shilong Liu
</p>
# Any2Poster: Any-Source Poster Generation Across Modalities and Domains

Any2Poster converts any input into a professional poster and evaluates that poster in real time.

## Any2Poster Example 😁

![aaai_poster6 preview](docs/assets/aaai_poster6.png)

Full poster PDF: [aaai_poster6.pdf](docs/assets/aaai_poster6.pdf)

## Install and setup

```bash
git clone https://anonymous.4open.science/r/Any2Poster-NeurIPS-2026
cd any2poster
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[render]"
playwright install chromium
```

Create a `.env` file in the project root by copying `.env.example` and filling in your keys.

## Get started

Simple run:

```bash
any2poster paper.pdf
```

Specify output path:

```bash
any2poster paper.pdf -o results/my_poster.pdf
```

Use a white and blue style:

```bash
any2poster paper.pdf --style arctic -o poster.pdf
```

Enable feedback loop and set a higher DPI:

```bash
any2poster paper.pdf --feedback --feedback-iters 2 --dpi 250 -o poster.pdf
```

Resume from cached checkpoints:

```bash
any2poster paper.pdf --resume -o poster.pdf
```

## Supported inputs

any2poster accepts:

- PDF
- DOCX
- PPTX
- LaTeX (.tex)
- Markdown or plain text
- Jupyter Notebook (.ipynb)
- HTML or URL
- Video (YouTube URL)

## Output and naming

The poster is saved as PDF and PNG and an HTML artifact is also produced next to the output path.

Examples:

```
any2poster paper.pdf           -> paper_poster.pdf
any2poster "aaai paper.pdf"    -> aaai_paper_poster.pdf
any2poster paper.pdf -o out.pdf -> out.pdf
```

If the output file already exists, the pipeline auto-increments the name.

## Color themes

Preset styles:

- arctic
- academic_gray
- periwinkle
- forest
- onyx
- steel_blue
- aquamarine
- ruby
- light
- dark (random from frozen_lake, neon_ice, bubblegum_dark)
- auto
- custom

Custom colors:

```bash
any2poster paper.pdf \
  --style custom \
  --bg-color "#F5F5F5" \
  --header-color "#1A237E" \
  --subheader-color "#3949AB" \
  --header-text-color "#FFFFFF" \
  --text-color "#1A1A1A"
```

Add logos:

```bash
any2poster paper.pdf \
  --logo-left path/to/conference_logo.png \
  --logo-right path/to/institution_logo.png
```

## CLI options

```
Usage: any2poster [OPTIONS] INPUT_PATH

Options:
  -o, --output PATH              Output file (.pdf or .png). Default: <name>_poster.pdf
  --style STYLE                  Color theme [default: light]
  --mode [professional|casual]   Layout mode [default: professional]
  --width FLOAT                  Poster width in inches [default: 48.0]
  --height FLOAT                 Poster height in inches [default: 36.0]
  --dpi INT                      Output resolution
  --logo-left PATH               Left banner logo
  --logo-right PATH              Right banner logo
  --resume / --no-resume         Resume from checkpoint
  --debug / --no-debug           Verbose output
  --generate-visuals             Generate AI diagrams
  --style-chain / --no-style-chain  Use first panel as style reference
  --llm-model TEXT               LLM model ID
  --image-model TEXT             Image model ID
  --vision-model TEXT            Vision model for quality checks
  --quality-check / --no-quality-check  Validate generated visuals
  --quality-retries INT          Extra retries when quality fails
  --feedback / --no-feedback     Panel feedback loop
  --feedback-iters INT           Max feedback iterations
  --feedback-model TEXT          Vision model for feedback loop
  --column-gap FLOAT             Column gap override in inches
  --interactive                  Prompt for style and mode
  --bg-color TEXT                Custom background hex
  --header-color TEXT            Custom banner hex
  --subheader-color TEXT         Custom panel header hex
  --header-text-color TEXT       Custom header text hex
  --text-color TEXT              Custom body text hex
```

## Evaluation

Run a full dataset sample with evaluation:

```bash
python scripts/run_p2p_first_eval.py --index 0 --run-all --feedback --feedback-iters 2 --disable-vlm-ppl
```

BenchQuiz on a single poster:

```bash
python -m any2poster.eval.paperquiz \
  --poster-image path/to/poster.png \
  --poster-html path/to/poster.html \
  --qa-json path/to/qa.json \
  --models "openai/gpt-4o" \
  --out eval_results/paper_id/any2poster/paperquiz.json
```

VLM as judge on a single poster:

```bash
python -m any2poster.eval.vlm_judge \
  --poster-image path/to/poster.png \
  --model "openai/gpt-4o" \
  --out eval_results/paper_id/any2poster/vlm_judge.json
```

Stats metrics on a single poster:

```bash
python -m any2poster.eval.stats \
  --paper-id paper_id \
  --method-name any2poster \
  --gen-poster-png path/to/poster.png \
  --gen-poster-html path/to/poster.html \
  --gt-poster-png path/to/gt.png \
  --disable-vlm-ppl
```


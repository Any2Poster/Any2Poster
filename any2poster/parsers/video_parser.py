"""Video parser — transcripts and metadata from video URLs and local video files.

Supported sources
-----------------
- YouTube URLs        (youtube-transcript-api → yt-dlp subtitles → Whisper)
- Vimeo / other URLs  (yt-dlp subtitles → Whisper)
- Local video files   (Whisper directly)

Segmentation strategy
---------------------
For research talks, temporal position in the video is a far stronger signal
than keyword matching alone.  A speaker at 40 % of a 90-minute talk is almost
certainly covering methods regardless of their exact word choices.

The pipeline therefore combines two signals:
  1. Time-proportional priors  — where in the talk we are
  2. Keyword boosts            — explicit topic markers push the label forward

Both signals are combined into a single score per section type, and progression
is still strictly monotonic (talks never go backward).

Keyframe extraction (optional)
------------------------------
When opencv-python is installed, the parser also extracts visually distinct
keyframes from the video and attaches them as ExtractedFigure objects.  These
flow naturally into the VLM analyse stage so slide diagrams and figures from
the talk appear in the generated poster.

  pip install opencv-python   # frame extraction
  pip install imagehash       # perceptual-hash deduplication (recommended)

Optional dependencies (install whichever are available)
-------------------------------------------------------
  pip install youtube-transcript-api   # fastest for YouTube; no download needed
  pip install yt-dlp                   # metadata + subtitle extraction + audio/video download
  pip install faster-whisper           # preferred transcription backend (CPU-friendly)
  pip install openai-whisper           # fallback transcription backend
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from any2poster.parsers.base import BaseParser
from any2poster.models import (
    ParsedDocument,
    ExtractedFigure,
    ExtractedSection,
    SectionType,
)


# ──────────────────────────────────────────────────────────────────────────────
# URL patterns
# ──────────────────────────────────────────────────────────────────────────────

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:"
    r"youtube\.com/watch\?.*?v=|"
    r"youtu\.be/|"
    r"youtube\.com/embed/|"
    r"youtube\.com/v/"
    r")([A-Za-z0-9_\-]{11})",
    re.IGNORECASE,
)
_VIMEO_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:vimeo\.com|player\.vimeo\.com/video)/(\d+)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation constants
# ──────────────────────────────────────────────────────────────────────────────

_PAUSE_THRESHOLD_S: float = 1.8
_TARGET_PARA_WORDS: int = 150   # smaller paragraphs → more section candidates
_MIN_SECTION_WORDS: int = 40

# Section order — strictly monotonic during labelling
_SECTION_ORDER: list[str] = [
    "introduction",
    "background",
    "methods",
    "experiments",
    "results",
    "conclusion",
]

# Hard time boundaries: a paragraph at time_frac gets the label of the last
# boundary it has passed.  This gives roughly equal-sized sections for any talk
# length without relying on the soft Gaussian overlap that causes everything to
# pile into "results".
_SECTION_BOUNDARIES: list[tuple[float, str]] = [
    (0.00, "introduction"),
    (0.15, "background"),
    (0.28, "methods"),
    (0.52, "experiments"),
    (0.72, "results"),
    (0.88, "conclusion"),
]

# Keyword signals — deliberately broad to catch conversational phrasing
_SECTION_SIGNALS: dict[str, list[str]] = {
    "introduction": [
        "i'm here to", "i'll share", "i'll talk about", "today i", "in this talk",
        "let me start", "let me begin", "i want to present", "we address",
        "the problem", "the challenge", "our goal", "our motivation",
        "why this matters", "the question", "contributions", "overview",
        "in this work", "in this paper", "we present", "we introduce",
    ],
    "background": [
        "related work", "prior work", "previous work", "background",
        "existing", "traditional", "state of the art", "literature",
        "conventional", "earlier work", "has been studied", "has been shown",
        "researchers have", "studies have", "prior research",
    ],
    "methods": [
        "our approach", "our method", "our system", "our model", "we design",
        "we propose", "the key idea", "the main idea", "the idea is",
        "our framework", "the algorithm", "specifically", "concretely",
        "the way we", "what we built", "how it works", "the design",
        "knowledge engineering", "interface design", "system design",
        "we developed", "we built", "we created", "our solution",
    ],
    "experiments": [
        "we evaluate", "we conducted", "we ran", "study", "user study",
        "experiment", "evaluation", "participants", "we recruited",
        "we collected", "dataset", "we tested", "our study",
        "we measured", "procedure", "we asked", "conditions",
    ],
    "results": [
        "result", "finding", "we found", "we observed", "we saw",
        "significantly", "improvement", "better", "worse", "accuracy",
        "performance", "score", "compared to", "outperform",
        "the data show", "the analysis", "figure shows", "table shows",
        "p value", "effect", "students", "participants showed",
    ],
    "conclusion": [
        "in conclusion", "to summarize", "in summary", "to conclude",
        "we showed", "we have shown", "future work", "limitation",
        "takeaway", "key insight", "to wrap up", "thank you",
        "open question", "next step", "moving forward", "overall",
        "what we learned", "implication",
    ],
}

_SECTION_TITLES: dict[str, str] = {
    "introduction": "Introduction",
    "background":   "Background & Related Work",
    "methods":      "Method",
    "experiments":  "Experimental Setup",
    "results":      "Results & Findings",
    "conclusion":   "Conclusion",
}

# Speech filler words and disfluencies to strip from the transcript
_FILLER_RE = re.compile(
    r"\b(?:uh+h*|um+|hmm+|hm+|mhm+|uh-?huh|uh-?uh|uhuh+|uhhuh+|um-hm|"
    r"you know|i mean|sort of|kind of|like i said|"
    r"right\??|okay\??|you see)\b",
    re.IGNORECASE,
)

# Noise patterns in captions (also strips >> speaker-turn markers)
_NOISE_RE = re.compile(
    r"\[.*?\]|\(.*?\)|<[^>]+>|&\w+;|>>\s*",
    re.IGNORECASE,
)

# Repeated word/phrase including contractions (e.g. "we we've", "that's that's")
_STUTTER_RE = re.compile(r"\b(\w[\w']*)\s+\1\b", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Internal data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _VideoMetadata:
    title: str = "Untitled Video"
    uploader: str = ""
    description: str = ""
    duration_s: int = 0
    platform: str = "unknown"


@dataclass
class _Segment:
    text: str
    start: float = 0.0
    duration: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class VideoParser(BaseParser):
    """Parse video URLs and local video files into poster-ready documents."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"]

    def parse(self, input_path: str, output_dir: Path) -> ParsedDocument:
        output_dir.mkdir(parents=True, exist_ok=True)
        is_url = input_path.startswith(("http://", "https://"))

        metadata = self._fetch_metadata(input_path, is_url)
        segments = self._get_transcript(input_path, is_url, output_dir)

        if not segments:
            raise ValueError(
                f"Could not extract a transcript from: {input_path}\n\n"
                "Install at least one transcript backend:\n"
                "  pip install youtube-transcript-api   (YouTube only, no download)\n"
                "  pip install yt-dlp                   (all platforms)\n"
                "  pip install faster-whisper           (audio transcription)\n"
                "  pip install openai-whisper           (audio transcription fallback)"
            )

        sections = self._build_sections(segments, metadata)
        figures = self._get_keyframes(input_path, is_url, segments, output_dir)

        full_text_parts: list[str] = [metadata.title]
        if metadata.uploader:
            full_text_parts.append(metadata.uploader)
        if metadata.description:
            full_text_parts.append(metadata.description[:600])
        full_text_parts += [s.content for s in sections if s.content]
        full_text = "\n\n".join(p for p in full_text_parts if p).strip()

        return ParsedDocument(
            source_path=input_path,
            source_format="video",
            title=metadata.title,
            authors=metadata.uploader,
            sections=sections,
            figures=figures,
            tables=[],
            raw_text=full_text,
            total_words=self._count_words(full_text),
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _fetch_metadata(self, input_path: str, is_url: bool) -> _VideoMetadata:
        if not is_url:
            return _VideoMetadata(
                title=Path(input_path).stem.replace("_", " ").replace("-", " ").title(),
                platform="local",
            )

        meta = _VideoMetadata(platform=self._detect_platform(input_path))

        # Try yt-dlp first (richest metadata)
        if self._fetch_metadata_ytdlp(input_path, meta):
            return meta

        # HTML page scraping fallback — no auth, no extra dependencies
        if meta.platform == "youtube":
            self._fetch_metadata_page(input_path, meta)

        return meta

    def _fetch_metadata_ytdlp(self, url: str, meta: _VideoMetadata) -> bool:
        try:
            import yt_dlp  # type: ignore
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
            if info:
                meta.title = (info.get("title") or meta.title).strip()
                meta.uploader = (info.get("uploader") or info.get("channel") or "").strip()
                raw_desc = (info.get("description") or "").strip()
                meta.description = raw_desc[:800] if raw_desc else ""
                meta.duration_s = int(info.get("duration") or 0)
                return True
        except Exception:
            pass
        return False

    def _fetch_metadata_page(self, url: str, meta: _VideoMetadata) -> None:
        """Scrape the YouTube watch page to extract title from embedded JSON."""
        try:
            import requests
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=15,
            )
            if r.status_code != 200:
                return
            # Prefer JSON-embedded title (more reliable, no HTML entity issues)
            m = re.search(r'"title"\s*:\s*"([^"]{3,200})"', r.text)
            if m:
                title = m.group(1).strip()
                if title and title.lower() not in ("youtube", ""):
                    meta.title = title
                    return
            # Fallback: <title> tag
            m = re.search(r"<title>([^<]+)</title>", r.text)
            if m:
                title = re.sub(r"\s*[-|]\s*YouTube\s*$", "", m.group(1)).strip()
                if title and title.lower() not in ("youtube", ""):
                    meta.title = title
            # Also try to extract channel/author name
            if not meta.uploader:
                for pat in (r'"author"\s*:\s*"([^"]{2,80})"', r'"channelName"\s*:\s*"([^"]{2,80})"'):
                    cm = re.search(pat, r.text)
                    if cm:
                        meta.uploader = cm.group(1).strip()
                        break
        except Exception:
            pass

    def _detect_platform(self, url: str) -> str:
        if _YOUTUBE_RE.search(url):
            return "youtube"
        if _VIMEO_RE.search(url):
            return "vimeo"
        return "other"

    # ── Transcript acquisition ─────────────────────────────────────────────────

    def _get_transcript(
        self, input_path: str, is_url: bool, output_dir: Path
    ) -> list[_Segment]:
        if not is_url:
            return self._whisper_transcribe(Path(input_path))

        if _YOUTUBE_RE.search(input_path):
            segments = self._youtube_transcript_api(input_path)
            if segments:
                return segments
            segments = self._yt_dlp_subtitles(input_path, output_dir)
            if segments:
                return segments

        audio_path = self._download_audio(input_path, output_dir)
        if audio_path:
            return self._whisper_transcribe(audio_path)

        return []

    def _youtube_transcript_api(self, url: str) -> list[_Segment]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
        except ImportError:
            return []

        match = _YOUTUBE_RE.search(url)
        if not match:
            return []
        video_id = match.group(1)

        api = YouTubeTranscriptApi()

        # Direct fetch — try English variants first (fastest path)
        for lang in (("en",), ("en-US",), ("en-GB",)):
            try:
                raw = api.fetch(video_id, languages=lang)
                return self._snippets_to_segments(raw)
            except Exception:
                pass

        # List all available transcripts and pick / translate the best one
        try:
            transcript_list = api.list(video_id)
            transcript = None

            try:
                transcript = transcript_list.find_manually_created_transcript(
                    ["en", "en-US", "en-GB"]
                )
            except Exception:
                pass

            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        ["en", "en-US", "en-GB"]
                    )
                except Exception:
                    pass

            if transcript is None:
                try:
                    transcript = next(iter(transcript_list))
                    lang_code = getattr(transcript, "language_code", "en")
                    if lang_code not in ("en", "en-US", "en-GB"):
                        transcript = transcript.translate("en")
                except Exception:
                    return []

            raw = transcript.fetch()
            return self._snippets_to_segments(raw)

        except Exception:
            return []

    def _snippets_to_segments(self, raw) -> list[_Segment]:
        segments: list[_Segment] = []
        for entry in raw:
            if isinstance(entry, dict):
                text = entry.get("text", "")
                start = float(entry.get("start", 0.0))
                duration = float(entry.get("duration", 0.0))
            else:
                text = str(getattr(entry, "text", ""))
                start = float(getattr(entry, "start", 0.0))
                duration = float(getattr(entry, "duration", 0.0))
            cleaned = self._clean_caption(text)
            if cleaned:
                segments.append(_Segment(text=cleaned, start=start, duration=duration))
        return segments

    def _yt_dlp_subtitles(self, url: str, output_dir: Path) -> list[_Segment]:
        try:
            import yt_dlp  # type: ignore
        except ImportError:
            return []

        sub_dir = output_dir / "_subtitles"
        sub_dir.mkdir(exist_ok=True)
        opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "skip_download": True,
            "outtmpl": str(sub_dir / "sub"),
            "subtitlesformat": "vtt",
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception:
            return []

        vtt_files = sorted(sub_dir.glob("*.vtt"))
        return self._parse_vtt(vtt_files[0]) if vtt_files else []

    def _parse_vtt(self, vtt_path: Path) -> list[_Segment]:
        try:
            text = vtt_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        _TIME_RE = re.compile(
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+"
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
        )
        segments: list[_Segment] = []
        seen: set[str] = set()

        for block in re.split(r"\n\n+", text):
            lines = block.strip().splitlines()
            time_match = None
            content_lines: list[str] = []
            for line in lines:
                m = _TIME_RE.match(line.strip())
                if m:
                    time_match = m
                elif line.strip() and not line.strip().isdigit() and "WEBVTT" not in line:
                    content_lines.append(line.strip())
            if not time_match or not content_lines:
                continue
            h, mi, s, ms = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3)), int(time_match.group(4))
            start = h * 3600 + mi * 60 + s + ms / 1000
            eh, emi, es, ems = int(time_match.group(5)), int(time_match.group(6)), int(time_match.group(7)), int(time_match.group(8))
            end = eh * 3600 + emi * 60 + es + ems / 1000
            raw = self._clean_caption(" ".join(content_lines))
            if raw and raw not in seen:
                seen.add(raw)
                segments.append(_Segment(text=raw, start=start, duration=end - start))
        return segments

    def _download_audio(self, url: str, output_dir: Path) -> Path | None:
        try:
            import yt_dlp  # type: ignore
        except ImportError:
            return None
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "audio.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception:
            return None
        for candidate in output_dir.glob("audio.*"):
            return candidate
        return None

    def _whisper_transcribe(self, audio_path: Path) -> list[_Segment]:
        if not audio_path.exists():
            return []
        result = self._faster_whisper(audio_path)
        if result is not None:
            return result
        result = self._openai_whisper(audio_path)
        return result if result is not None else []

    def _faster_whisper(self, audio_path: Path) -> list[_Segment] | None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            return None
        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segs, _ = model.transcribe(str(audio_path), beam_size=5, language="en")
            return [_Segment(text=s.text.strip(), start=s.start, duration=s.end - s.start) for s in segs if s.text.strip()]
        except Exception:
            return None

    def _openai_whisper(self, audio_path: Path) -> list[_Segment] | None:
        try:
            import whisper  # type: ignore
        except ImportError:
            return None
        try:
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), language="en", verbose=False)
            return [
                _Segment(text=s["text"].strip(), start=float(s.get("start", 0.0)), duration=float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                for s in result.get("segments", []) if s.get("text", "").strip()
            ]
        except Exception:
            return None

    # ── Text cleaning ─────────────────────────────────────────────────────────

    def _clean_caption(self, text: str) -> str:
        text = _NOISE_RE.sub(" ", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_prose(self, text: str) -> str:
        """Deep cleaning for poster-ready prose: remove fillers, stutters, normalise."""
        text = _FILLER_RE.sub("", text)
        text = _STUTTER_RE.sub(r"\1", text)
        text = _STUTTER_RE.sub(r"\1", text)  # second pass handles triples
        # Trim sentences that start mid-thought (common at transcript boundaries)
        text = re.sub(r"^\s*[a-z][^.!?]*?(?=[A-Z])", "", text)
        # Collapse multiple spaces and fix spacing around punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s([,.!?])", r"\1", text)
        text = self._normalize_text(text.strip())
        return text

    # ── Keyframe extraction ───────────────────────────────────────────────────

    def _get_keyframes(
        self,
        input_path: str,
        is_url: bool,
        segments: list[_Segment],
        output_dir: Path,
    ) -> list[ExtractedFigure]:
        """Dispatch keyframe extraction — skip entirely if opencv is missing."""
        try:
            import cv2  # type: ignore  # noqa: F401
        except ImportError:
            return []

        if not is_url:
            return self._extract_keyframes(Path(input_path), segments, output_dir)

        video_path = self._download_video(input_path, output_dir)
        if not video_path:
            return []
        try:
            return self._extract_keyframes(video_path, segments, output_dir)
        finally:
            try:
                video_path.unlink()
            except Exception:
                pass

    def _download_video(self, url: str, output_dir: Path) -> Path | None:
        """Download lowest-quality video stream for keyframe extraction."""
        try:
            import yt_dlp  # type: ignore
        except ImportError:
            return None

        opts = {
            "quiet": True,
            "no_warnings": True,
            # Prefer a pre-merged mp4 at ≤480p; fall back to anything available
            "format": "best[height<=480][ext=mp4]/best[height<=480]/worst[ext=mp4]/worst",
            "outtmpl": str(output_dir / "video_kf.%(ext)s"),
            "merge_output_format": "mp4",
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception:
            return None

        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            candidate = output_dir / f"video_kf{ext}"
            if candidate.exists():
                return candidate
        for candidate in output_dir.glob("video_kf.*"):
            return candidate
        return None

    def _extract_keyframes(
        self,
        video_path: Path,
        segments: list[_Segment],
        output_dir: Path,
        *,
        interval_s: float = 30.0,
        max_frames: int = 12,
        hash_threshold: int = 8,
    ) -> list[ExtractedFigure]:
        """Sample frames every interval_s, deduplicate via perceptual hash, keep max_frames."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return []

        try:
            import imagehash  # type: ignore
            from PIL import Image as _PILImage  # type: ignore
            _has_imagehash = True
        except ImportError:
            _has_imagehash = False

        frames_dir = output_dir / "keyframes"
        frames_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frame_count / fps if fps > 0 else 0.0

        frame_step = max(1, int(fps * interval_s))
        # list of (timestamp_s, frame_bgr)
        sampled: list[tuple[float, object]] = []

        frame_idx = 0
        while frame_idx < total_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            sampled.append((frame_idx / fps, frame))
            frame_idx += frame_step

        cap.release()

        if not sampled:
            return []

        # Perceptual-hash deduplication — keep frames that differ visually
        if _has_imagehash:
            kept: list[tuple[float, object]] = []
            kept_hashes: list[object] = []
            for ts, frame in sampled:
                pil = _PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                h = imagehash.phash(pil)
                if not kept_hashes or min(abs(h - kh) for kh in kept_hashes) > hash_threshold:
                    kept.append((ts, frame))
                    kept_hashes.append(h)
        else:
            kept = sampled

        # Evenly sub-sample down to max_frames preserving temporal spread
        if len(kept) > max_frames:
            step = len(kept) / max_frames
            kept = [kept[int(i * step)] for i in range(max_frames)]

        figures: list[ExtractedFigure] = []
        for i, (ts, frame) in enumerate(kept):
            frame_path = frames_dir / f"keyframe_{i + 1:03d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            mins = int(ts) // 60
            secs = int(ts) % 60
            section_label = self._section_label_at(ts, duration_s)
            caption = f"Video keyframe at {mins}:{secs:02d}"
            if section_label:
                caption += f" — {section_label}"

            figures.append(ExtractedFigure(
                figure_id=f"keyframe_{i + 1:03d}",
                caption=caption,
                path=frame_path,
            ))

        return figures

    def _section_label_at(self, timestamp_s: float, duration_s: float) -> str:
        """Return the human-readable section title for a given timestamp."""
        if duration_s <= 0:
            return ""
        frac = min(timestamp_s / duration_s, 1.0)
        label = "introduction"
        for boundary_frac, sec_name in _SECTION_BOUNDARIES:
            if frac >= boundary_frac:
                label = sec_name
        return _SECTION_TITLES.get(label, label.title())

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _build_sections(
        self, segments: list[_Segment], metadata: _VideoMetadata
    ) -> list[ExtractedSection]:
        paragraphs = self._segments_to_paragraphs(segments)
        labelled = self._label_paragraphs(paragraphs, segments)
        sections = self._merge_into_sections(labelled, metadata)
        return sections

    def _segments_to_paragraphs(
        self, segments: list[_Segment]
    ) -> list[tuple[str, float]]:
        if not segments:
            return []

        has_timing = sum(1 for s in segments if s.start > 0.01) > len(segments) // 4
        paragraphs: list[tuple[str, float]] = []
        current_texts: list[str] = []
        current_start = segments[0].start

        for i, seg in enumerate(segments):
            current_texts.append(seg.text)
            word_count = sum(len(t.split()) for t in current_texts)

            gap_after = 0.0
            if has_timing and i + 1 < len(segments):
                gap_after = segments[i + 1].start - (seg.start + seg.duration)

            split_on_pause = has_timing and gap_after > _PAUSE_THRESHOLD_S
            split_on_length = word_count >= _TARGET_PARA_WORDS
            is_last = i == len(segments) - 1

            if split_on_pause or split_on_length or is_last:
                text = self._clean_prose(" ".join(current_texts))
                if text:
                    paragraphs.append((text, current_start))
                current_texts = []
                if i + 1 < len(segments):
                    current_start = segments[i + 1].start

        return paragraphs

    def _label_paragraphs(
        self,
        paragraphs: list[tuple[str, float]],
        segments: list[_Segment],
    ) -> list[tuple[str, float, str]]:
        """Label each paragraph using hard time boundaries + optional keyword advance.

        Hard boundaries give each section a fair, roughly proportional slice of the
        talk regardless of keyword density.  Keywords can advance the label by one
        step when they strongly favour the next section.  A final monotonic pass
        ensures labels never go backward.
        """
        if not paragraphs:
            return []

        total_duration = max(
            (s.start + s.duration for s in segments if s.start > 0), default=0.0
        )
        use_time = total_duration > 30.0
        n = len(paragraphs)

        raw: list[tuple[str, float, str]] = []
        for para_idx, (text, start) in enumerate(paragraphs):
            lower = text.lower()

            if use_time and total_duration > 0:
                time_frac = min(start / total_duration, 1.0)
            else:
                time_frac = para_idx / max(n - 1, 1)

            # Hard boundary: last threshold we've passed
            base_label = "introduction"
            for boundary_frac, sec_name in _SECTION_BOUNDARIES:
                if time_frac >= boundary_frac:
                    base_label = sec_name

            base_idx = _SECTION_ORDER.index(base_label)

            # One-step keyword look-ahead: advance if next section's keywords clearly win
            if base_idx < len(_SECTION_ORDER) - 1:
                next_sec = _SECTION_ORDER[base_idx + 1]
                next_kw = sum(1 for sig in _SECTION_SIGNALS[next_sec] if sig in lower)
                curr_kw = sum(1 for sig in _SECTION_SIGNALS[base_label] if sig in lower)
                if next_kw >= curr_kw + 2:
                    base_label = next_sec

            raw.append((text, start, base_label))

        # Monotonic pass: labels may only move forward
        min_idx = 0
        labelled: list[tuple[str, float, str]] = []
        for text, start, label in raw:
            idx = _SECTION_ORDER.index(label)
            if idx < min_idx:
                label = _SECTION_ORDER[min_idx]
            else:
                min_idx = idx
            labelled.append((text, start, label))

        return labelled

    def _merge_into_sections(
        self,
        labelled: list[tuple[str, float, str]],
        metadata: _VideoMetadata,
    ) -> list[ExtractedSection]:
        if not labelled:
            return []

        groups: list[tuple[str, list[str]]] = []
        current_label = labelled[0][2]
        current_texts: list[str] = []

        for text, _start, label in labelled:
            if label == current_label:
                current_texts.append(text)
            else:
                groups.append((current_label, list(current_texts)))
                current_label = label
                current_texts = [text]
        groups.append((current_label, current_texts))

        sections: list[ExtractedSection] = []
        seen_label_count: dict[str, int] = {}
        section_counter = 0

        for label, texts in groups:
            content = " ".join(texts).strip()
            word_count = self._count_words(content)

            if word_count < _MIN_SECTION_WORDS:
                if sections:
                    prev = sections[-1]
                    prev.content = (prev.content + " " + content).strip()
                    prev.word_count = self._count_words(prev.content)
                continue

            seen_label_count[label] = seen_label_count.get(label, 0) + 1
            count = seen_label_count[label]
            section_counter += 1

            base_title = _SECTION_TITLES.get(label, label.title())
            title = base_title if count == 1 else f"{base_title} (cont.)"
            section_type = SectionType(label) if label in SectionType._value2member_map_ else SectionType.OTHER

            sections.append(
                ExtractedSection(
                    section_id=f"section_{section_counter}",
                    title=title,
                    section_type=section_type,
                    content=content,
                    level=1,
                    word_count=word_count,
                )
            )

        # If description is substantive, prepend as abstract
        if metadata.description and self._count_words(metadata.description) >= 25:
            abstract = ExtractedSection(
                section_id="section_0",
                title="Abstract",
                section_type=SectionType.ABSTRACT,
                content=metadata.description.strip(),
                level=1,
                word_count=self._count_words(metadata.description),
            )
            sections.insert(0, abstract)
            for i, sec in enumerate(sections):
                sec.section_id = f"section_{i + 1}"

        return sections

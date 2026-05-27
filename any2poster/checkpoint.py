# Manages pipeline checkpoints for save, load, and resume.

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from any2poster.models import PosterConfig

T = TypeVar("T", bound=BaseModel)

_STAGE_ORDER = ["parse", "chunk", "analyze", "plan", "generate"]


def get_checkpoint_dir(config: "PosterConfig") -> Path:
    d = Path(config.checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_panels_dir(config: "PosterConfig") -> Path:
    d = get_checkpoint_dir(config) / "panels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir(config: "PosterConfig") -> Path:
    d = get_checkpoint_dir(config) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(config: "PosterConfig", stage: str, data: BaseModel) -> Path:
    cp_dir = get_checkpoint_dir(config)
    path = cp_dir / f"{stage}.json"
    path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_checkpoint(config: "PosterConfig", stage: str, model_class: type[T]) -> T | None:
    cp_dir = get_checkpoint_dir(config)
    path = cp_dir / f"{stage}.json"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return model_class.model_validate_json(raw)
    except Exception:
        return None


def should_run_stage(config: "PosterConfig", stage: str) -> bool:
    if not config.resume:
        return True
    cp_dir = get_checkpoint_dir(config)
    path = cp_dir / f"{stage}.json"
    return not path.exists()


def clear_checkpoints(config: "PosterConfig", from_stage: str | None = None) -> None:
    cp_dir = get_checkpoint_dir(config)
    if from_stage is None:
        stages_to_clear = _STAGE_ORDER
    else:
        try:
            idx = _STAGE_ORDER.index(from_stage)
            stages_to_clear = _STAGE_ORDER[idx:]
        except ValueError:
            stages_to_clear = [from_stage]
    for stage in stages_to_clear:
        path = cp_dir / f"{stage}.json"
        if path.exists():
            path.unlink()
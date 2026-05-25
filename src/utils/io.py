from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def specimen_id_from_render(render_path: Path, root_dir: Path | None = None) -> str:
    path_for_id = render_path.relative_to(root_dir) if root_dir is not None else render_path
    name = path_for_id.stem
    base = name.rsplit("_view", 1)[0] if "_view" in name else name
    rel_parent = path_for_id.parent
    return base if str(rel_parent) == "." else str(rel_parent / base)


def group_renders_by_specimen(render_files: Iterable[Path], root_dir: Path | None = None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for fp in render_files:
        sid = specimen_id_from_render(fp, root_dir=root_dir)
        grouped.setdefault(sid, []).append(fp)
    for sid in grouped:
        grouped[sid] = sorted(grouped[sid])
    return grouped


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

"""Run path bundle with one source of stage directory names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidvrd_auto.utils.paths import repo_root


_STAGES = {
    "inputs": "inputs",
    "vocabulary": "vocabulary",
    "detect": "detect",
    "track": "track",
    "track_qc": "track_qc",
    "rule": "rule",
    "semantic": "semantic",
    "merge": "merge",
    "global": "global",
    "verify": "verify",
    "export": "export",
}


@dataclass(frozen=True)
class VideoPaths:
    repo_dir: Path
    run_dir: Path
    video_dir: Path
    video_id: str

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        if not video_id or video_id in {".", ".."} or Path(video_id).name != video_id:
            raise ValueError("video_id must be a non-empty path-safe name")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "repo_dir", Path(self.repo_dir).expanduser().resolve())
        object.__setattr__(self, "run_dir", Path(self.run_dir).expanduser().resolve())
        object.__setattr__(self, "video_dir", Path(self.video_dir).expanduser().resolve())

    @classmethod
    def for_video(
        cls,
        run_dir: Path | str,
        video_id: str,
        *,
        repo_dir: Path | str | None = None,
    ) -> VideoPaths:
        root = Path(repo_dir).expanduser().resolve() if repo_dir is not None else repo_root()
        run = Path(run_dir).expanduser()
        if not run.is_absolute():
            run = root / run
        run = run.resolve()
        return cls(repo_dir=root, run_dir=run, video_dir=run / "videos" / video_id, video_id=video_id)

    def stage(self, name: str) -> Path:
        try:
            folder = _STAGES[name]
        except KeyError as exc:
            choices = ", ".join(sorted(_STAGES))
            raise KeyError(f"unknown stage '{name}'; expected one of: {choices}") from exc
        return self.video_dir / folder

    def artifact(self, stage: str, name: str) -> Path:
        filename = Path(name)
        if filename.name != str(name) or str(name) in {"", ".", ".."}:
            raise ValueError("artifact name must be a single filename")
        return self.stage(stage) / filename

    @property
    def inputs_dir(self) -> Path:
        return self.stage("inputs")

    @property
    def vocabulary_dir(self) -> Path:
        return self.stage("vocabulary")

    @property
    def detect_dir(self) -> Path:
        return self.stage("detect")

    @property
    def track_dir(self) -> Path:
        return self.stage("track")

    @property
    def track_qc_dir(self) -> Path:
        return self.stage("track_qc")

    @property
    def rule_dir(self) -> Path:
        return self.stage("rule")

    @property
    def semantic_dir(self) -> Path:
        return self.stage("semantic")

    @property
    def merge_dir(self) -> Path:
        return self.stage("merge")

    @property
    def global_dir(self) -> Path:
        return self.stage("global")

    @property
    def verify_dir(self) -> Path:
        return self.stage("verify")

    @property
    def export_dir(self) -> Path:
        return self.stage("export")

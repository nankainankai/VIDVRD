from __future__ import annotations

"""Shared trajectory discontinuity handling before relation inference."""

from collections.abc import Mapping, Sequence


def split_frames_on_track_jumps(
    frames: Sequence[int], track_ids: Sequence[int], jump_frames: Mapping[str, Sequence[int]] | None
) -> list[list[int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    jumps = {
        int(frame)
        for track_id in track_ids
        for frame in (jump_frames or {}).get(str(track_id), [])
    }
    groups: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if any(groups[-1][-1] < jump <= frame for jump in jumps):
            groups.append([])
        groups[-1].append(frame)
    return groups

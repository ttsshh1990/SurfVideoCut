#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified SD card ingest for SoloShot tracks.

Layout on SD card (example):
  /Volumes/SD/
    Track_2025_08_08_063354/
      SS3_TRACK_VIDEO_2025_08_08_063424-1.mp4
      SS3_TRACK_VIDEO_2025_08_08_063424-2.mp4
      ...
    Track_2025_08_10_090259/
      SS3_TRACK_VIDEO_2025_08_10_090321-1.mp4
      ...

What this does
--------------
- Scans --sd-root for folders named "Track_*".
- For each Track folder, finds video parts named "SS3_TRACK_VIDEO_*.mp4".
- Sorts parts by the numeric suffix after the last dash (e.g. -1, -2, ...).
- Concatenates them (stream copy) into a single MP4 in --outdir named <TrackFolder>.mp4.
- On success, removes the original Track folder (unless --keep is provided).

Usage
-----
python sd_ingest.py --sd-root /Volumes/SD --outdir ~/Videos/Surf/clips

Options:
  --ffmpeg / --ffprobe  : paths to tools (default: ffmpeg/ffprobe in PATH)
  --verify              : ffprobe duration before & after concat
  --dry-run             : print what would happen, do nothing
  --keep                : do NOT delete Track_* folders after success

Notes
-----
- Uses ffmpeg concat demuxer with -c copy (no re-encode).
- Expects all parts in a Track to have compatible codecs/params (camera default).
"""
from __future__ import annotations
import argparse
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

# -----------------------------
# Utilities
# -----------------------------

def run(cmd: List[str], dry_run: bool=False) -> int:
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return proc.returncode


def ffprobe_duration(ffprobe: str, path: Path, dry_run: bool=False) -> Optional[float]:
    if dry_run:
        return None
    try:
        out = subprocess.check_output([
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ], stderr=subprocess.STDOUT).decode("utf-8", "ignore").strip()
        return float(out) if out else None
    except Exception:
        return None


def human_time(sec: Optional[float]) -> str:
    if sec is None:
        return "?"
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def write_concat_list(files: List[Path], concat_file: Path):
    def esc(p: Path) -> str:
        return str(p).replace("'", "'\\''")
    lines = [f"file '{esc(p)}'" for p in files]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

# -----------------------------
# Core
# -----------------------------

TRACK_DIR_RE = re.compile(r"^Track_\d{4}_\d{2}_\d{2}_\d{6}$")
PART_RE = re.compile(r"^SS3_TRACK_VIDEO_\d{4}_\d{2}_\d{2}_\d{6}-(\d+)\.mp4$", re.IGNORECASE)


def find_track_dirs(sd_root: Path) -> List[Path]:
    return sorted([p for p in sd_root.iterdir() if p.is_dir() and TRACK_DIR_RE.match(p.name)])


def find_and_sort_parts(track_dir: Path) -> List[Path]:
    parts = []
    for p in sorted(track_dir.iterdir()):
        if not p.is_file():
            continue
        m = PART_RE.match(p.name)
        if m:
            idx = int(m.group(1))
            parts.append((idx, p))
    parts.sort(key=lambda t: t[0])
    return [p for _, p in parts]


def combine_track(ffmpeg: str, parts: List[Path], out_path: Path, dry_run: bool=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = out_path.with_suffix(out_path.suffix + ".concat.txt")
    write_concat_list(parts, concat_list)
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_path)
    ]
    run(cmd, dry_run=dry_run)
    # Do not delete the concat file automatically; keep for provenance

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Combine SoloShot Track_* folders into single MP4s and optionally delete the folders.")
    ap.add_argument('--sd-root', required=True, help='Root path containing Track_* folders')
    ap.add_argument('--outdir', required=True, help='Destination directory for combined MP4s')
    ap.add_argument('--ffmpeg', default='ffmpeg', help='Path to ffmpeg')
    ap.add_argument('--ffprobe', default='ffprobe', help='Path to ffprobe')
    ap.add_argument('--verify', action='store_true', help='ffprobe durations before/after combine')
    ap.add_argument('--keep', action='store_true', help='Do NOT delete Track_* folders after successful combine')
    ap.add_argument('--dry-run', action='store_true', help='Print actions only, make no changes')

    args = ap.parse_args()

    sd_root = Path(args.sd_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not sd_root.exists():
        raise SystemExit(f"SD root not found: {sd_root}")

    tracks = find_track_dirs(sd_root)
    if not tracks:
        print("[INFO] No Track_* folders found.")
        return

    print(f"[INFO] Found {len(tracks)} Track folders under {sd_root}")

    for track in tracks:
        print(f"\n[TRACK] {track.name}")
        parts = find_and_sort_parts(track)
        if not parts:
            print("  [WARN] No SS3_TRACK_VIDEO_*.mp4 parts found, skipping")
            continue
        for p in parts:
            print("   part:", p.name)

        out_mp4 = outdir / f"{track.name}.mp4"
        if out_mp4.exists() and not args.dry_run:
            print(f"  [SKIP] Output already exists: {out_mp4}")
            continue

        # Optional verification: total input duration
        if args.verify:
            tin = sum((ffprobe_duration(args.ffprobe, p, dry_run=args.dry_run) or 0.0) for p in parts)
            print(f"  [VERIFY] Total input duration ≈ {human_time(tin)} ({tin:.2f}s)")

        # Combine
        combine_track(args.ffmpeg, parts, out_mp4, dry_run=args.dry_run)
        print(f"  [OK] Wrote {out_mp4}")

        # Optional verification: output duration
        if args.verify:
            tout = ffprobe_duration(args.ffprobe, out_mp4, dry_run=args.dry_run)
            print(f"  [VERIFY] Output duration ≈ {human_time(tout)} ({tout if tout is not None else '?':.2f}s)")

        # Cleanup
        if not args.keep:
            if args.dry_run:
                print(f"  [CLEAN] Would remove folder: {track}")
            else:
                print(f"  [CLEAN] Removing folder: {track}")
                shutil.rmtree(track, ignore_errors=True)
        else:
            print("  [KEEP] Track folder preserved")

    print("\n[DONE] SD ingest complete.")


if __name__ == '__main__':
    main()
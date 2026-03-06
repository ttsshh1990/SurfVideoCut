#!/usr/bin/env python3
"""
Combine all .mp4 parts in a folder into one MP4 using ffmpeg concat demuxer.

Designed for SoloShot dumps where parts are directly inside a folder
(e.g., SS3_...-1.MP4, SS3_...-2.MP4, Track_..._part_01.mp4, etc.).

Usage:
  python3 combine_simple.py --sd-root /path/to/folder --outdir /path/to/output
    [--ffmpeg ffmpeg] [--dry-run]

Behavior:
- Finds all *.mp4 files directly under --sd-root (non-recursive).
- Sorts by numeric suffix if present (e.g., -1, _part_01); otherwise by name.
- Writes output as <outdir>/<sd_root_basename>.mp4.
- Always keeps the source folder (no deletion).
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args():
    ap = argparse.ArgumentParser(description="Combine mp4 parts in a folder into a single mp4 (no re-encode).")
    ap.add_argument("--sd-root", required=True, help="Folder containing mp4 parts")
    ap.add_argument("--outdir", required=True, help="Destination folder for combined mp4")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg (default: ffmpeg in PATH)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not run ffmpeg")
    return ap.parse_args()


def probe_duration(ffmpeg: str, path: Path) -> float:
    try:
        out = subprocess.check_output([
            ffmpeg, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ], stderr=subprocess.STDOUT).decode("utf-8", "ignore").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


SUFFIX_RE = re.compile(r"(?:[_-]part[_-]?|[-_])(\d+)", re.IGNORECASE)


def find_parts(sd_root: Path) -> List[Path]:
    return sorted([p for p in sd_root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"], key=sort_key)


def sort_key(path: Path):
    m = SUFFIX_RE.search(path.stem)
    if m:
        try:
            return (int(m.group(1)), path.name)
        except Exception:
            pass
    return (10**9, path.name)


def write_concat(parts: List[Path], dest: Path):
    lines = []
    for p in parts:
        esc = str(p).replace("'", "'\\''")
        lines.append(f"file '{esc}'")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffmpeg(ffmpeg: str, concat_file: Path, out_path: Path, total_duration: float, dry_run: bool = False) -> int:
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy",
        "-progress", "-", "-nostats",
        str(out_path)
    ]
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    out_ms = float(line.split("=")[1])
                    sec = out_ms / 1_000_000.0
                    if total_duration > 0:
                        pct = max(0.0, min(1.0, sec / total_duration))
                        print(f"[PROGRESS] {pct:.4f}")
                except Exception:
                    pass
    except KeyboardInterrupt:
        proc.kill()
        raise
    proc.wait()
    return proc.returncode


def main():
    args = parse_args()
    sd_root = Path(args.sd_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not sd_root.exists() or not sd_root.is_dir():
        print(f"[ERR] sd-root not found or not a directory: {sd_root}")
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    parts = find_parts(sd_root)
    if not parts:
        print(f"[ERR] No .mp4 parts found in {sd_root}")
        return 1

    stem = sd_root.name
    out_path = outdir / f"{stem}.mp4"
    concat_file = out_path.with_suffix(out_path.suffix + ".concat.txt")
    write_concat(parts, concat_file)

    total_dur = sum(probe_duration(args.ffmpeg, p) or 0.0 for p in parts)
    code = run_ffmpeg(args.ffmpeg, concat_file, out_path, total_duration=total_dur, dry_run=args.dry_run)
    try:
        concat_file.unlink()
    except Exception:
        pass

    if code != 0:
        print(f"[ERR] ffmpeg failed with code {code}")
        return code

    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

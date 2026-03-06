#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast clipper for macOS (M1/M2/M3 friendly): cut multiple segments via ffmpeg -c copy
with optional parallel jobs to better use Apple Silicon cores.

Examples
--------
# From a text file (one start-end per line, supports mm:ss or hh:mm:ss)
python cut_segments_parallel.py --input surf.mp4 --outdir clips \
  --segments-file segments.txt --jobs 4

# From CLI list
python cut_segments_parallel.py --input surf.mp4 --outdir clips \
  --segments "6:22-6:25,7:33-7:36,7:47-7:50,8:06-8:10,8:23-8:27" --jobs 4

Notes
-----
- Uses fast seek (`-ss` before `-i`) + `-c copy` → no re-encode, near-keyframe accuracy.
- Drops subtitles & data streams by default (`-sn -dn`).
- Maps first video stream and (optionally) first audio stream; no heavy probing.
- Runs multiple ffmpeg processes concurrently up to --jobs.
"""

import argparse
import concurrent.futures as futures
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_timecode(tc: str) -> float:
    parts = tc.strip().split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    elif len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Invalid timecode: {tc}")


def parse_segments(seg_str: str) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    for token in seg_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid segment: {token}")
        a, b = token.split("-", 1)
        segs.append((parse_timecode(a), parse_timecode(b)))
    return segs


def parse_segments_file(file_path: Path) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "-" not in line:
                raise ValueError(f"Invalid segment line: {line}")
            a, b = line.split("-", 1)
            segs.append((parse_timecode(a), parse_timecode(b)))
    return segs


def cut_one(input_path: str, output_path: str, start_s: float, end_s: float, keep_audio: bool) -> Tuple[str, bool, str]:
    duration = max(0.01, end_s - start_s)
    # Map first video stream and (optionally) first audio stream; '?' makes it optional.
    maps = ["-map", "0:v:0", "-map", "0:a:0?"] if keep_audio else ["-map", "0:v:0"]
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        *maps,
        "-c", "copy",
        "-sn", "-dn",
        str(output_path),
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ok = (r.returncode == 0) and Path(output_path).exists() and Path(output_path).stat().st_size > 0
        return str(output_path), ok, ("" if ok else r.stderr[-400:])
    except Exception as e:
        return str(output_path), False, str(e)


def main():
    p = argparse.ArgumentParser(description="Cut multiple segments via ffmpeg -c copy with parallel jobs")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--outdir", default="clips", help="Output directory")
    p.add_argument("--segments", help="Comma-separated 'start-end,...' (mm:ss or hh:mm:ss)")
    p.add_argument("--segments-file", help="Text file with one 'start-end' per line")
    p.add_argument("--jobs", type=int, default=1, help="Parallel ffmpeg processes (try 4 on M2)")
    p.add_argument("--keep-audio", action="store_true", help="Include first audio track (default off)")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Build segment list
    segs: List[Tuple[float, float]] = []
    if args.segments:
        segs.extend(parse_segments(args.segments))
    if args.segments_file:
        segs.extend(parse_segments_file(Path(args.segments_file)))
    if not segs:
        raise ValueError("Provide --segments and/or --segments-file")

    # De-dup & sort (optional)
    segs = sorted(set(segs), key=lambda x: (x[0], x[1]))

    tasks = []
    for i, (s, e) in enumerate(segs, start=1):
        outp = Path(args.outdir) / f"{inp.stem}_part_{i:02d}{inp.suffix}"
        tasks.append((s, e, outp))

    # Run in parallel
    worked = 0
    with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(cut_one, str(inp), str(outp), s, e, args.keep_audio) for (s, e, outp) in tasks]
        for fut in futures.as_completed(futs):
            out_path, ok, err = fut.result()
            if ok:
                worked += 1
                print(f"[OK] {out_path}")
            else:
                print(f"[ERR] {out_path}: {err}")

    print(f"Done: {worked}/{len(tasks)} segments exported → {args.outdir}")


if __name__ == "__main__":
    main()

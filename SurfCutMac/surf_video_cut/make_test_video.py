#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a test video (or multiple parts) from selected segments (with padding)
and remap a golden file from the original timeline to the test video’s timeline.

Now supports:
  --max-length N   Split the test set into <=N-minute parts, writing
                   outdir/clips1, outdir/clips2, ... each with its own
                   test video and remapped golden.

Inputs
------
- --input: original long video
- --segments: detector's previous result on the original timeline (start-end[ \t index])
- --golden: hand-edited ground truth on the original timeline (start-end[ \t index])
- --pad-sec: seconds to add before/after each detected segment
- --merge-gap: merge padded segments within this gap
- --max-length: maximum length per output test part (minutes); omit to produce a single test video
- --outdir: where to write outputs
- --ffmpeg/--ffprobe/--ffmpeg-args: tools and extras

Outputs
-------
If --max-length NOT given:
  outdir/<stem>_test.mp4
  outdir/<stem>_test_golden.txt
  outdir/<stem>_test_map.tsv

If --max-length is given (parts):
  outdir/clips1/<stem>_test_part01.mp4
  outdir/clips1/<stem>_test_part01_golden.txt
  outdir/clips1/<stem>_test_part01_map.tsv
  outdir/clips2/...
"""

from __future__ import annotations
import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

# ----------------- Timecode helpers -----------------

def parse_tc(tc: str) -> float:
    tc = tc.strip()
    parts = tc.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(tc)

def fmt_tc(sec: float) -> str:
    if sec < 0: sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h * 3600 + m * 60)
    if h > 0:
        return f"{h}:{m:02d}:{int(round(s)):02d}" if abs(s - round(s)) < 1e-3 else f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{int(round(s)):02d}" if abs(s - round(s)) < 1e-3 else f"{m}:{s:05.2f}"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ----------------- File I/O helpers -----------------

def read_segments(path: Path) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    if not path.exists():
        return segs
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()[0]  # first token must be start-end
        if "-" not in tok:
            continue
        a, b = tok.split("-", 1)
        try:
            sa, sb = parse_tc(a), parse_tc(b)
        except Exception:
            continue
        if sb > sa:
            segs.append((sa, sb))
    segs.sort()
    return segs

def write_segments(path: Path, segs: List[Tuple[float, float]]):
    lines = [f"{fmt_tc(s)}-{fmt_tc(e)}" for s, e in segs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ----------------- Interval ops -----------------

def merge_intervals(segs: List[Tuple[float, float]], gap: float = 0.0) -> List[Tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged: List[Tuple[float, float]] = []
    cs, ce = segs[0]
    for s, e in segs[1:]:
        if s - ce <= gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def intersect(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return (s, e) if e > s else None

# ----------------- FFmpeg helpers -----------------

def ffprobe_duration(ffprobe: str, path: Path) -> Optional[float]:
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

def write_ffconcat_for_chunks(src: Path, chunks: List[Tuple[float, float]], concat_file: Path):
    """
    ffconcat v1.0 file using repeated 'file' entries and inpoint/outpoint.
    """
    lines = ["ffconcat version 1.0"]
    esc = lambda p: str(p).replace("'", "'\\''")
    for (s, e) in chunks:
        lines.append(f"file '{esc(src)}'")
        lines.append(f"inpoint {max(0.0, s):.6f}")
        lines.append(f"outpoint {max(0.0, e):.6f}")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_ffmpeg_concat(ffmpeg: str, concat_file: Path, out_mp4: Path, extra: str = ""):
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-safe", "0",
        "-f", "concat",
        "-i", str(concat_file),
        "-c", "copy",
    ]
    if extra:
        cmd += shlex.split(extra)
    cmd += [str(out_mp4)]
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed with code {proc.returncode}")

# ----------------- Remapping logic -----------------

def build_dst_mapping(chunks: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """
    For each source chunk [s,e), compute destination offset on test video:
      dst_start = cumulative duration of previous chunks
    Returns list of (src_s, src_e, dst_start).
    """
    mapping: List[Tuple[float, float, float]] = []
    acc = 0.0
    for s, e in chunks:
        mapping.append((s, e, acc))
        acc += (e - s)
    return mapping

def remap_golden_to_test(golden: List[Tuple[float, float]], mapping: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for gs, ge in golden:
        for cs, ce, dsoff in mapping:
            ov = intersect((gs, ge), (cs, ce))
            if ov:
                os, oe = ov
                ds = dsoff + (os - cs)
                de = dsoff + (oe - cs)
                out.append((ds, de))
    out = merge_intervals(out, gap=1e-6)
    return out

def split_test_into_parts(total_len: float, part_len: float) -> List[Tuple[float, float]]:
    """
    Split test timeline [0, total_len) into [a,b) parts of length <= part_len.
    """
    parts = []
    t = 0.0
    while t < total_len - 1e-9:
        end = min(total_len, t + part_len)
        parts.append((t, end))
        t = end
    return parts

def dest_window_to_source_slices(mapping: List[Tuple[float, float, float]],
                                 dw: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Given a destination window [ds,de) on the test timeline, invert mapping to
    source slices [(ss,se), ...] that produce this window.
    """
    ds, de = dw
    out: List[Tuple[float, float]] = []
    for cs, ce, dsoff in mapping:
        seg_dst = (dsoff, dsoff + (ce - cs))
        ov = intersect(seg_dst, (ds, de))
        if not ov:
            continue
        ods, ode = ov  # overlap on dest axis
        # map back to source
        ss = cs + (ods - dsoff)
        se = cs + (ode - dsoff)
        if se > ss:
            out.append((ss, se))
    return out

def remap_golden_part(remapped_golden_on_test: List[Tuple[float, float]],
                      part_window: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Given golden segments on the full test timeline, cut to part window and shift to local [0, part_len).
    """
    ds, de = part_window
    out: List[Tuple[float, float]] = []
    for gs, ge in remapped_golden_on_test:
        ov = intersect((gs, ge), (ds, de))
        if ov:
            s, e = ov
            out.append((s - ds, e - ds))
    out = merge_intervals(out, gap=1e-6)
    return out

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Create test video from detected segments (with padding); optional split into <=N-minute parts and remap golden.")
    ap.add_argument('--input', required=True, help='Original long video')
    ap.add_argument('--segments', required=True, help='Detector result on original timeline (start-end [index])')
    ap.add_argument('--golden', required=True, help='Golden truth on original timeline (start-end [index])')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--pad-sec', type=float, default=1.5, help='Pad seconds before/after each detected segment')
    ap.add_argument('--merge-gap', type=float, default=0.0, help='Merge padded chunks separated by <= this gap')
    ap.add_argument('--max-length', type=float, default=None, help='Max length per test part in MINUTES (omit to produce a single test video)')
    ap.add_argument('--ffmpeg', default='ffmpeg', help='Path to ffmpeg')
    ap.add_argument('--ffprobe', default='ffprobe', help='Path to ffprobe')
    ap.add_argument('--ffmpeg-args', default='', help='Extra args to ffmpeg (e.g. \"-movflags +faststart\")')
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    duration = ffprobe_duration(args.ffprobe, src)
    if duration is None:
        raise SystemExit("Could not read input duration via ffprobe")

    # 1) Load & pad detector segments; clamp and merge
    det = read_segments(Path(args.segments).expanduser().resolve())
    if not det:
        raise SystemExit(f"No segments found in {args.segments}")

    padded: List[Tuple[float, float]] = []
    for s, e in det:
        ps = clamp(s - args.pad_sec, 0.0, duration)
        pe = clamp(e + args.pad_sec, 0.0, duration)
        if pe > ps:
            padded.append((ps, pe))
    padded = merge_intervals(padded, gap=args.merge_gap)
    if not padded:
        raise SystemExit("No segments after padding/merge.")

    stem = src.stem

    # Build the test mapping (source -> contiguous test)
    mapping = build_dst_mapping(padded)
    total_test_len = sum(e - s for s, e in padded)

    # Remap GOLDEN to full test timeline (once)
    golden = read_segments(Path(args.golden).expanduser().resolve())
    if not golden:
        raise SystemExit(f"No golden segments found in {args.golden}")
    test_golden_full = remap_golden_to_test(golden, mapping)

    # If not splitting, produce single test video
    if not args.max_length:
        concat_file = outdir / f"{stem}_test.concat.ffconcat"
        out_mp4 = outdir / f"{stem}_test.mp4"
        write_ffconcat_for_chunks(src, padded, concat_file)
        run_ffmpeg_concat(args.ffmpeg, concat_file, out_mp4, extra=args.ffmpeg_args)
        print(f"[OK] Test video written: {out_mp4}")

        # Write full remapped golden
        test_golden = outdir / f"{stem}_test_golden.txt"
        write_segments(test_golden, test_golden_full)
        print(f"[OK] Remapped golden written: {test_golden} ({len(test_golden_full)} segments)")

        # Also write a mapping table for debugging
        map_tsv = outdir / f"{stem}_test_map.tsv"
        with open(map_tsv, 'w', encoding='utf-8') as f:
            f.write("src_start\tsrc_end\tdst_start\tdst_end\n")
            acc = 0.0
            for s, e in padded:
                f.write(f"{s:.3f}\t{e:.3f}\t{acc:.3f}\t{acc + (e - s):.3f}\n")
                acc += (e - s)
        print(f"[OK] Timeline map written: {map_tsv}")
        return

    # -------------- Split into parts --------------
    part_len = float(args.max_length) * 60.0
    parts = split_test_into_parts(total_test_len, part_len)
    if not parts:
        print("[WARN] No parts produced; nothing to write.")
        return

    # For each part, compute source slices, write concat, build video, cut golden to local part
    for idx, (pd_s, pd_e) in enumerate(parts, start=1):
        # Invert mapping: get the source [ss,se) slices that produce this dest window
        src_slices = dest_window_to_source_slices(mapping, (pd_s, pd_e))
        if not src_slices:
            continue

        subdir = outdir / f"clips{idx}"
        subdir.mkdir(parents=True, exist_ok=True)

        part_name = f"{stem}_test_part{idx:02d}"
        concat_file = subdir / f"{part_name}.concat.ffconcat"
        out_mp4 = subdir / f"{part_name}.mp4"

        write_ffconcat_for_chunks(src, src_slices, concat_file)
        run_ffmpeg_concat(args.ffmpeg, concat_file, out_mp4, extra=args.ffmpeg_args)
        print(f"[OK] Wrote part #{idx}: {out_mp4}  (len≈{pd_e - pd_s:.2f}s)")

        # Part-local golden (cut to [pd_s, pd_e) and shift to [0, part_len))
        golden_local = remap_golden_part(test_golden_full, (pd_s, pd_e))
        golden_file = subdir / f"{part_name}_golden.txt"
        write_segments(golden_file, golden_local)
        print(f"[OK] Remapped golden for part #{idx}: {golden_file} ({len(golden_local)} segs)")

        # Mapping table for this part
        map_tsv = subdir / f"{part_name}_map.tsv"
        with open(map_tsv, 'w', encoding='utf-8') as f:
            f.write("src_start\tsrc_end\tdst_start(local)\tdst_end(local)\n")
            # Recompute local mapping rows for debug
            # (src slices listed in order; accumulate local offsets)
            acc = 0.0
            for ss, se in src_slices:
                f.write(f"{ss:.3f}\t{se:.3f}\t{acc:.3f}\t{acc + (se - ss):.3f}\n")
                acc += (se - ss)

    print(f"[DONE] Wrote {len(parts)} part(s) under {outdir}/clips*/")

if __name__ == "__main__":
    main()
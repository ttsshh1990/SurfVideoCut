#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-all pipeline:
1) (Optional) SD ingest via sd_ingest.py (combine Track_* -> outdir/Track_*.mp4)
2) Detect (process.py) to generate <video_stem>_segments.txt
3) Cut (cut_video.py) to create clips from the segments

All artifacts live under **--outdir**:
- Combined video:            outdir/<video>.mp4
- Segments file:             outdir/<video>_segments.txt  (now: 'start-end<TAB>index')
- Debug frames (detect):     outdir/debug/...
- Clips:                     outdir/<video>/clip_###.mp4

New: --rerun-cut
----------------
Edit the segments file in --outdir (you may delete lines).
Rerun with --rerun-cut to:
- clean old clips,
- reindex remaining lines (indices become consecutive),
- cut again from the updated file.

New: --ingest-all
-----------------
Use the simplified sd_ingest.py to scan --sd-root for Track_* folders, combine each to
outdir/Track_*.mp4, and then run detect+cut for each combined file automatically.
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List, Optional, Tuple

# -----------------------------
# Helpers
# -----------------------------

def run(cmd: List[str], dry_run: bool=False, cwd: Optional[Path]=None) -> int:
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return proc.returncode

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cleanup_old_clips(outdir: Path, stem: str,
                      combined_path: Optional[Path]=None,
                      segments_file: Optional[Path]=None,
                      dry_run: bool=False):
    """
    Remove previous clips for a given video stem.
    Preferred layout: outdir/<stem>/... (nuke the folder).
    Fallback: flat files like <stem>_clip_*.* in outdir.
    Never deletes the combined video or segments file.
    """
    def _safe(p: Path) -> bool:
        if combined_path and p.resolve() == combined_path.resolve():
            return False
        if segments_file and p.resolve() == segments_file.resolve():
            return False
        return True

    removed = 0

    # Preferred: per-stem folder
    cand_dirs = [outdir / stem, outdir / f"{stem}_clips"]
    for d in cand_dirs:
        if d.exists() and d.is_dir():
            print(f"[CLEAN] Removing folder: {d}")
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
            removed += 1

    # Fallback: flat files if no folder existed
    if removed == 0:
        patterns = [
            f"{stem}_clip_*",
            f"{stem}_clip_*.mp4", f"{stem}_clip_*.mov", f"{stem}_clip_*.m4v",
            f"{stem}_clip_*.mkv", f"{stem}_clip_*.avi", f"{stem}_clip_*.mpg",
        ]
        seen = set()
        for pat in patterns:
            for p in outdir.glob(pat):
                if p in seen:
                    continue
                seen.add(p)
                if not _safe(p):
                    continue
                if p.is_file():
                    print(f"[CLEAN] Removing file: {p}")
                    if not dry_run:
                        try:
                            p.unlink()
                        except Exception as e:
                            print(f"[WARN] Failed to delete {p}: {e}")
                    removed += 1

    if removed == 0:
        print(f"[CLEAN] No existing clips found for stem '{stem}' in {outdir}")

# ---------- Segments IO (readable with indices) ----------

def _parse_segment_line(line: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """
    Accepts lines like:
      'mm:ss-mm:ss'                (no index)
      'mm:ss-mm:ss\t12'           (with index)
      'mm:ss.s-mm:ss.s   7'        (spaces also OK)
    Returns (start, end, index or None), or None for invalid/comment/empty.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # Split into time-range and optional index token
    parts = s.split()
    if len(parts) == 0:
        return None
    time_range = parts[0]
    idx = None
    if len(parts) >= 2:
        # try to parse trailing token as int
        try:
            idx = int(parts[-1])
        except ValueError:
            idx = None
    if "-" not in time_range:
        return None
    a, b = time_range.split("-", 1)
    a, b = a.strip(), b.strip()
    if not a or not b:
        return None
    return (a, b, idx)

def read_segments_file(path: Path) -> List[Tuple[str, str]]:
    """Read segments allowing optional trailing index; return [(start_str, end_str), ...]."""
    out: List[Tuple[str, str]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_segment_line(line)
        if parsed is None:
            continue
        a, b, _ = parsed
        out.append((a, b))
    return out

def write_segments_with_indices(path: Path, segments: List[Tuple[str, str]]):
    """Write segments as 'start-end<TAB>index' with 1-based consecutive indices."""
    lines = []
    for i, (a, b) in enumerate(segments, start=1):
        lines.append(f"{a}-{b}\t{i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Segments reindexed & saved: {path} ({len(segments)} clips)")

def write_segments_times_only(path: Path, segments: List[Tuple[str, str]]):
    """Write segments as 'start-end' only (for cut_video.py consumption)."""
    lines = [f"{a}-{b}" for (a, b) in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='SD combine (via sd_ingest.py) -> detect (process.py) -> cut (cut_video.py)')

    # Inputs / outputs
    ap.add_argument('--sd-root', type=str, default=None, help='SD root containing Track_* folders (e.g., /Volumes/SD)')
    ap.add_argument('--combined', type=str, default='session.mp4', help='(legacy) Single combined output name (not used by Track_* ingest)')
    ap.add_argument('--input', type=str, default=None, help='Skip ingest and use this local input video directly')
    ap.add_argument('--outdir', type=str, required=True, help='Directory for combined video(s), segments, debug and clips')

    # Ingest mode (simplified Track_* flow)
    ap.add_argument('--ingest-all', action='store_true', help='Combine ALL Track_* folders under --sd-root into outdir/Track_*.mp4, then process each')
    ap.add_argument('--keep-tracks', action='store_true', help='Do NOT delete Track_* folders after successful combine (passed to sd_ingest.py)')
    ap.add_argument('--verify-ingest', action='store_true', help='Verify durations before/after combine (ffprobe in sd_ingest.py)')

    # Tool paths
    ap.add_argument('--python', type=str, default=sys.executable, help='Python interpreter to run child scripts')
    ap.add_argument('--process-script', type=str, default='process_improved.py', help='Detector script filename')
    ap.add_argument('--clip-script', type=str, default='cut_video.py', help='Clipper script filename')
    ap.add_argument('--sd-ingest-script', type=str, default='sd_ingest.py', help='Path to sd_ingest.py')

    # sd_ingest legacy passthrough (kept only for tool path flags)
    ap.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg (for sd_ingest)')
    ap.add_argument('--ffprobe', type=str, default='ffprobe', help='Path to ffprobe (for sd_ingest)')

    # process.py options (wire the most common flags; extras via --extra-process-args)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--frame-stride', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--vx-thresh', type=float, default=60.0)
    ap.add_argument('--min-seg-sec', type=float, default=0.8)
    ap.add_argument('--merge-gap-sec', type=float, default=1.0)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--near-px', type=int, default=80)
    ap.add_argument('--iou-thresh', type=float, default=0.05)
    ap.add_argument('--preroll', type=float, default=1.5)
    ap.add_argument('--postroll', type=float, default=1.0)
    ap.add_argument('--smooth', action='store_true')
    
    # NEW: Activity classifier options (for process_improved.py)
    ap.add_argument('--activity-threshold', type=float, default=0.3, help='Activity classification threshold (lower = more sensitive to surfing)')
    ap.add_argument('--activity-model', type=str, default='models/activity_classifier.pth', help='Path to trained activity classifier')
    ap.add_argument('--no-velocity-fallback', action='store_true', help='Disable velocity fallback if classifier fails')
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda','mps'], help='Device for detection and classification')

    # cut_video.py options
    ap.add_argument('--jobs', type=int, default=4, help='Parallel workers for clip cutting')

    # Passthrough extras
    ap.add_argument('--extra-process-args', type=str, default='', help='Quoted extra args for process.py (e.g. "--device mps --save-debug")')
    ap.add_argument('--extra-cut-args', type=str, default='', help='Quoted extra args for cut_video.py')

    # Control
    ap.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    ap.add_argument('--rerun-cut', action='store_true', help='Skip detection, clear old clips, and re-cut using edited segments file')

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    # Helper: run detect + cut for a single video
    def run_detect_and_cut(combined_path: Path):
        if not combined_path.exists() and not args.dry_run:
            raise SystemExit(f"Combined input not found: {combined_path}")
        combined_stem = combined_path.stem
        segments_file = outdir / f"{combined_stem}_segments.txt"

        # Detect
        process_cmd = [
            args.python, args.process_script,
            '--input', str(combined_path),
            '--outdir', str(outdir),
            '--imgsz', str(args.imgsz),
            '--frame-stride', str(args.frame_stride),
            '--batch-size', str(args.batch_size),
            '--vx-thresh', str(args.vx_thresh),
            '--min-seg-sec', str(args.min_seg_sec),
            '--merge-gap-sec', str(args.merge_gap_sec),
            '--conf', str(args.conf),
            '--near-px', str(args.near_px),
            '--iou-thresh', str(args.iou_thresh),
            '--preroll', str(args.preroll),
            '--postroll', str(args.postroll),
        ]
        if args.smooth:
            process_cmd.append('--smooth')
        
        # Add activity classifier parameters (for process_improved.py)
        process_cmd.extend([
            '--device', str(args.device),
            '--activity-threshold', str(args.activity_threshold),
            '--activity-model', str(args.activity_model),
        ])
        if args.no_velocity_fallback:
            process_cmd.append('--no-velocity-fallback')
        if args.extra_process_args:
            process_cmd += shlex.split(args.extra_process_args)
        run(process_cmd, dry_run=args.dry_run)

        # Reindex segments for readability (and to support deletion)
        segs = read_segments_file(segments_file) if segments_file.exists() else []
        if not segs and not args.dry_run:
            candidates = sorted(outdir.glob(f"{combined_stem}*_segments.txt"))
            if candidates:
                segments_file = candidates[-1]
                segs = read_segments_file(segments_file)
        write_segments_with_indices(segments_file, segs)
        print(f"[INFO] Using segments file: {segments_file}")

        # Cut into per-stem folder
        clips_outdir = outdir / combined_stem
        clips_outdir.mkdir(parents=True, exist_ok=True)

        # Cutter uses times-only file
        segs_for_cut = outdir / f"{combined_stem}_segments.__cut__.txt"
        write_segments_times_only(segs_for_cut, segs)

        clip_cmd = [
            args.python, args.clip_script,
            '--input', str(combined_path),
            '--outdir', str(clips_outdir),
            '--segments-file', str(segs_for_cut),
            '--jobs', str(args.jobs),
        ]
        if args.extra_cut_args:
            clip_cmd += shlex.split(args.extra_cut_args)
        run(clip_cmd, dry_run=args.dry_run)

        # clean temp
        try:
            if segs_for_cut.exists():
                segs_for_cut.unlink()
        except Exception:
            pass

    # Utility: pick newest Track_*.mp4 in outdir
    def newest_track_mp4(outdir: Path) -> Optional[Path]:
        candidates = list(outdir.glob('Track_*.mp4'))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    # ---------------- RERUN-CUT PATH ----------------
    if args.rerun_cut:
        if args.input:
            combined_path = Path(args.input).expanduser().resolve()
        else:
            # Auto-pick newest Track_*.mp4 in outdir if input not provided
            combined_path = newest_track_mp4(outdir)
            if combined_path:
                print(f"[RERUN] Auto-selected newest combined video: {combined_path.name}")
        if (combined_path is None or not combined_path.exists()) and not args.dry_run:
            raise SystemExit("--rerun-cut: provide --input or ensure a Track_*.mp4 exists in --outdir.")
        combined_stem = combined_path.stem
        segments_file = outdir / f"{combined_stem}_segments.txt"
        if not segments_file.exists():
            candidates = sorted(outdir.glob(f"{combined_stem}*_segments.txt"))
            if candidates:
                segments_file = candidates[-1]
        if not segments_file.exists() and not args.dry_run:
            raise SystemExit(f"--rerun-cut: segments file not found in {outdir} for stem '{combined_stem}'.")

        print(f"[RERUN] Using video:    {combined_path}")
        print(f"[RERUN] Using segments: {segments_file}")

        # Read, reindex (handles deletions), write back with indices
        segs = read_segments_file(segments_file)
        write_segments_with_indices(segments_file, segs)

        # Clean old clips then cut (into per-stem folder)
        clips_outdir = outdir / combined_stem
        cleanup_old_clips(outdir, combined_stem,
                          combined_path=combined_path,
                          segments_file=segments_file,
                          dry_run=args.dry_run)
        clips_outdir.mkdir(parents=True, exist_ok=True)

        # For cutter, use a temp stripped file (times only)
        segs_for_cut = outdir / f"{combined_stem}_segments.__cut__.txt"
        write_segments_times_only(segs_for_cut, segs)

        clip_cmd = [
            args.python, args.clip_script,
            '--input', str(combined_path),
            '--outdir', str(clips_outdir),
            '--segments-file', str(segs_for_cut),
            '--jobs', str(args.jobs),
        ]
        if args.extra_cut_args:
            clip_cmd += shlex.split(args.extra_cut_args)
        run(clip_cmd, dry_run=args.dry_run)

        # Clean up temp
        try:
            if segs_for_cut.exists():
                segs_for_cut.unlink()
        except Exception:
            pass

        print('[DONE] Rerun cut finished.')
        return

    # ---------------- FIRST-RUN / FULL PIPELINE ----------------
    # 1) Ingest ALL tracks (batch) -> detect+cut for each
    if args.ingest_all:
        if not args.sd_root:
            raise SystemExit('--sd-root is required when using --ingest-all')
        sd_root = Path(args.sd_root).expanduser().resolve()
        if not sd_root.exists():
            raise SystemExit(f'SD root not found: {sd_root}')

        # Run sd_ingest.py once (creates outdir/Track_*.mp4; removes Track_* unless --keep-tracks)
        ingest_cmd = [
            args.python, args.sd_ingest_script,
            '--sd-root', str(sd_root),
            '--outdir', str(outdir),
            '--ffmpeg', args.ffmpeg, '--ffprobe', args.ffprobe,
        ]
        if args.verify_ingest:
            ingest_cmd.append('--verify')
        if args.keep_tracks:
            ingest_cmd.append('--keep')
        run(ingest_cmd, dry_run=args.dry_run)

        # Iterate each combined Track_*.mp4 and run detect+cut
        combined_files = sorted(outdir.glob('Track_*.mp4'))
        if not combined_files and not args.dry_run:
            raise SystemExit(f"No combined Track_*.mp4 found in {outdir} after ingest.")
        for combined_path in combined_files:
            print(f"\n[PIPELINE] Processing {combined_path.name}")
            run_detect_and_cut(combined_path)
        print('\n[DONE] Batch pipeline finished for all Track_* videos.')
        return

    # 1b) Non-ingest-all: still use sd_ingest Track_* interface, then pick one file
    combined_files: List[Path] = []
    if args.input:
        combined_files = [Path(args.input).expanduser().resolve()]
        print(f"[INFO] Using existing input: {combined_files[0]}")
    else:
        if not args.sd_root:
            raise SystemExit('--sd-root is required when --input is not provided')
        sd_root = Path(args.sd_root).expanduser().resolve()
        if not sd_root.exists():
            raise SystemExit(f'SD root not found: {sd_root}')

        ingest_cmd = [
            args.python, args.sd_ingest_script,
            '--sd-root', str(sd_root),
            '--outdir', str(outdir),
            '--ffmpeg', args.ffmpeg, '--ffprobe', args.ffprobe,
        ]
        # make behavior mirror ingest_all but only continue with one file
        run(ingest_cmd, dry_run=args.dry_run)

        combined_files = sorted(outdir.glob('Track_*.mp4'))
        if not combined_files and not args.dry_run:
            raise SystemExit(f"No combined Track_*.mp4 found in {outdir} after ingest.")
        if len(combined_files) == 1:
            combined_path = combined_files[0]
            print(f"[INFO] Found exactly one combined file: {combined_path.name}")
        else:
            # pick newest by mtime; user can override via --input or --ingest-all
            combined_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            combined_path = combined_files[0]
            print(f"[WARN] Multiple Track_*.mp4 found; auto-selecting newest: {combined_path.name}")
            print("      Tip: pass --input <file> or use --ingest-all to process all.")
        # process the chosen one
        run_detect_and_cut(combined_path)
        print('[DONE] Pipeline finished.')
        return

    # If we got here and args.input was provided, process that single file
    run_detect_and_cut(combined_files[0])
    print('[DONE] Pipeline finished.')
    

if __name__ == '__main__':
    main()
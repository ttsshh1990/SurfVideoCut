# Surf Video Cut App - Design Plan

Goal: Local tool to manage four workflows:
- W1: SD -> combine -> clean -> detection -> manual edit -> export.
- W2: SD -> combine -> clean -> manual edit -> export (no detection).
- W3: Open existing combined MP4 -> manual edit -> export.
- W4: Open existing combined MP4 -> detection -> manual edit -> export.

Guiding principles:
- Reuse existing scripts: sd_ingest.py (combine/clean), process_improved.py (detection), cut_video.py (export).
- Local-only, no network; prefer Apple Silicon acceleration when available.
- Keep session artifacts together (combined MP4, segments, edits, exports).
- Edits always possible even without detection results.

Backend plan (FastAPI/Starlette or similar):
- /ingest (POST): sd_root, outdir, keep/delete tracks -> runs sd_ingest.py; returns combined_path(s), logs.
- /detect (POST): input_path, outdir, detection params -> runs process_improved.py; returns segments file path + parsed segments.
- /load (POST): input_path, optional segments_path -> loads video metadata + segments (empty if missing).
- /segments (GET/POST): in-memory + persisted segments JSON alongside text file; supports add/delete/trim/update order.
- /export (POST): input_path, segments list, outdir, jobs, keep_audio -> runs cut_video.py; returns per-clip status.
- /video (GET): range-serving of input video for the player.
- Session store: small JSON per session (e.g., session_meta.json) under outdir to track current video, params, and edits.

Frontend plan (single-page web UI):
- Source step:
  - Option A: "Combine from SD" (fields: SD root, keep/delete Track folders toggle). Button "Combine & Clean". On success, ask "Run detection now?".
  - Option B: "Open combined video" (file picker). If segments file found nearby, auto-load; else start with empty list; offer "Run detection now?".
- Detect step (optional):
  - Presets: High Recall (activity 0.1, conf 0.2, smooth on), Balanced (activity 0.15, conf 0.25), High Precision (activity 0.2, conf 0.25, min_seg 1.2).
  - Controls: device (auto/mps/cuda/cpu), activity threshold slider (0.05–0.3), conf, smooth toggle, merge gap, preroll/postroll, frame stride, imgsz.
  - Progress display (frames processed / ETA) using process output polling.
- Review/Edit step:
  - Video player with timeline bands for segments; click band to play that segment.
  - Clip table: start, end, duration; actions: play, delete, trim in/out via UI handles or numeric inputs; add new clip from current playhead range; reorder optional.
  - Save edits: writes segments JSON + segments.txt (times only) + segments indexed file (for readability).
- Export step:
  - Fields: output clips folder name (default: <stem>_<date>), keep-audio toggle, jobs slider.
  - Button "Export clips"; show per-clip status/progress.

Data/layout conventions:
- Session outdir: chosen by user (e.g., ~/Videos/SurfSessions/<date>).
- Combined video: outdir/<stem>.mp4 (from sd_ingest).
- Segments files:
  - <stem>_segments.txt (times only for cutter).
  - <stem>_segments_indexed.txt (with tab-separated indices for human edits; matches run_all style).
  - segments.json (authoritative list for UI with metadata).
- Debug (optional): outdir/debug/* from detection when enabled.
- Clips export: outdir/<stem>/<stem>_part_##.mp4 (same naming as cut_video.py).

Detection parameter defaults:
- device: auto (mps > cuda > cpu).
- imgsz: 640, frame_stride: 5, batch_size: 32, conf: 0.25, iou_thresh: 0.05, near_px: 80, min_seg_sec: 0.8, merge_gap_sec: 1.0, preroll: 1.5, postroll: 1.0, final_merge_gap_sec: 2.0.
- Activity threshold: 0.15 balanced; presets expose 0.1 (recall) and 0.2 (precision).
- Smooth: optional; smooth_gap defaults to merge_gap_sec.

Editing behavior:
- Always allow manual creation/trim/deletion regardless of detection.
- Timeline bands reflect current segments after every edit.
- Undo/redo (nice-to-have) via in-memory history; minimum viable: manual edits saved explicitly.

Export behavior:
- Use cut_video.py with segments.txt (times only) to avoid index parsing.
- Parallel jobs default 4 on Apple Silicon; keep-audio off by default.
- Surface errors per clip in UI.

Safety/cleanup:
- Combine step offers "delete Track folders after success" (default on).
- Keep concat list files from sd_ingest for provenance.
- No destructive deletion beyond Track folder cleanup.

UI mock status (ui_mock.html):
- No sidebar or session bar; single-column stack.
- Source card at top with inline buttons: SD root (Load/Open), MP4 (Open), output folder (Open), segments (Open); action buttons: Run all, Run detection, Clean, Export.
- Detection options moved into a collapsible details inside Source; presets plus advanced sliders.
- Review/Edit emphasized: large video area (~460px tall), timeline bands below; clips table narrowed with only # and Play/Delete actions (start/end/duration/trim removed).
- Export block removed from UI mock; export action remains in Source actions.

Next implementation steps:
- Scaffold backend service with the routes above; wire to existing scripts via subprocess.
- Implement frontend: use ui_mock.html structure; wire actions to backend endpoints; manage state (loaded video, segments, detection params) in JS with persistence (segments.json + txt).
- Add segment persistence and export integration; hook Clean button to sd_ingest cleanup only.

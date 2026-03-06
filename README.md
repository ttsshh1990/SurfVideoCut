# SurfCutMac

SurfCutMac is a macOS app for the surf video workflow:

- load an SD card session folder
- combine split MP4 files into one video
- review and manually edit clip segments
- export selected clips
- optionally run Python-based surf detection

## Current status

Working now:

- open a combined MP4 directly
- manually add, delete, and trim clips
- zoom and edit on the timeline
- combine split MP4 files from an SD session folder
- clean the selected SD session folder
- export clips into `OUTPUT_DIR/clips/`
- run detection if the Python ML dependencies are installed

Still not implemented:

- `Run All`
- opening an existing segment file from the UI

## Option 1: Download and run the app directly

This is feasible, but the compiled app should be distributed through GitHub Releases, not committed into the repo.

When a release is available:

1. Open the repo's `Releases` page.
2. Download the latest `SurfCutMac.zip`.
3. Unzip it.
4. Move `SurfCutMac.app` into `/Applications` if you want.
5. Launch it with right-click -> `Open` the first time.

Releases can be produced either:

- manually by building and zipping the app
- automatically by pushing a git tag like `v0.1.0`

Notes:

- `ffmpeg` is bundled inside the app.
- `python3` is still required on the Mac for `Combine` and `Export`.
- Detection additionally requires Python ML packages such as `torch`, `torchvision`, `opencv-python`, `ultralytics`, and `tqdm`.
- If `python3` is missing, the app will warn at startup.

## Option 2: Build locally from source

### Requirements

- macOS
- Xcode
- `python3` available in `PATH`

Optional for detection:

- Python packages required by `process_improved.py`

### Build in Xcode

1. Open [SurfCutMac.xcodeproj](/Users/tiantian/Documents/SoloShotEditor/SurfCutMac/SurfCutMac.xcodeproj) in Xcode.
2. Select the `SurfCutMac` scheme.
3. Build and run.

### Build from terminal

```bash
xcodebuild \
  -project SurfCutMac.xcodeproj \
  -scheme SurfCutMac \
  -configuration Debug \
  -derivedDataPath .deriveddata \
  build CODE_SIGNING_ALLOWED=NO
```

The built app will be located at:

```text
.deriveddata/Build/Products/Debug/SurfCutMac.app
```

## Runtime dependency model

### Included in the app bundle

- UI files
- Python scripts under `SurfCutMac/surf_video_cut/`
- YOLO model weights used by detection
- bundled `ffmpeg`

### Expected on the host Mac

- `python3` for `Combine` and `Export`
- Python ML packages for `Detection`

## Export output

Exports use the selected output folder as the session base folder.

- clips go into `OUTPUT_DIR/clips/`
- the segment file stays in the selected output folder

## Maintainer: create a downloadable app zip

Build the app, then package it:

```bash
cd SurfCutMac
xcodebuild \
  -project SurfCutMac.xcodeproj \
  -scheme SurfCutMac \
  -configuration Release \
  -derivedDataPath .deriveddata \
  build
ditto -c -k --sequesterRsrc --keepParent \
  .deriveddata/Build/Products/Release/SurfCutMac.app \
  SurfCutMac.zip
```

Upload `SurfCutMac.zip` to a GitHub Release.

## Maintainer: automated GitHub release

This repo includes a GitHub Actions workflow that:

- builds `SurfCutMac.app` on macOS
- zips it as `SurfCutMac.zip`
- uploads it as a workflow artifact
- publishes it to GitHub Releases when you push a tag like `v0.1.0`

To create a downloadable release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

After the workflow finishes, download the app from the GitHub Release asset.

## Repo layout

- [SurfCutMac](/Users/tiantian/Documents/SoloShotEditor/SurfCutMac/SurfCutMac): app source
- [SurfCutMac.xcodeproj](/Users/tiantian/Documents/SoloShotEditor/SurfCutMac/SurfCutMac.xcodeproj): Xcode project
- [SurfCutMac/surf_video_cut](/Users/tiantian/Documents/SoloShotEditor/SurfCutMac/SurfCutMac/surf_video_cut): bundled Python scripts and models

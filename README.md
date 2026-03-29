<!--
waldo - image region of interest tracker
Copyright (C) 2026 notweerdmonk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Waldo

Version: `1.1.0`

`waldo` tracks a moving region of interest across either a folder of image frames, a video file, or piped frame data on `stdin`.
It is packaged as a distributable Python module and includes `waldo.py` as a wrapper entrypoint.

## Current Status

- The tracker core is implemented and usable as version `1.1.0`.
- Phase 2 stdin pipeline support is implemented for ffmpeg-style raw `bgr24` and `image2pipe` PNG/JPEG streams.
- Packaging is PEP 517-first through `pyproject.toml`, with `setup.py` retained as a compatibility shim for older setuptools-based tooling.
- The PEP 517 workflow uses `pep517_backend.py` as the local build backend shim so setuptools wheel/sdist finalization can fall back cleanly when this environment raises `EXDEV` on `rename`.
- Runtime and development/build dependencies are split between `requirements.txt` and `requirements-dev.txt`.
- Installation and release guidance are documented in `INSTALL`.
- Release tarball contents are controlled explicitly through `MANIFEST.in`.
- Local build and install work in this environment through the documented wheel-build-then-install fallback, because direct `pip install .` can still fail in pip's wheel-cache finalization.

## Installation

```bash
.venv/bin/pip install -r requirements.txt
```

Development/build toolchain:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

PEP 517 build/install in this environment:

```bash
TMPDIR=/home/weerdmonk/Projects/waldo/tmp .venv/bin/python -m build --no-isolation
.venv/bin/pip install --no-deps dist/waldo-1.1.0-py3-none-any.whl
```

This flow uses `pyproject.toml` together with `pep517_backend.py`. The backend shim is part of the source distribution and is required for the documented local workaround in environments where backend artifact finalization can hit `EXDEV`.

Equivalent helper:

```bash
./scripts/install_local_pep517.sh
```

Compatibility check for older tooling:

```bash
.venv/bin/python setup.py --version
```

## Usage

Track from a frames folder using a template image:

```bash
.venv/bin/waldo \
  --frames-dir /path/to/frames \
  --template /path/to/template.png \
  --output-csv tracks.csv \
  --debug-dir debug_frames \
  --debug-every 10
```

Track from a video using a first-frame bounding box:

```bash
.venv/bin/waldo \
  --video /path/to/video.mp4 \
  --init-bbox 120,80,240,90 \
  --output-csv tracks.csv
```

Track from an ffmpeg rawvideo pipeline:

```bash
ffmpeg -i /path/to/video.mp4 -f rawvideo -pix_fmt bgr24 pipe:1 | \
.venv/bin/waldo \
  --stdin-format raw-bgr24 \
  --stdin-size 1366x680 \
  --template /path/to/template.png \
  --output-csv tracks.csv
```

Track from an ffmpeg `image2pipe` PNG pipeline:

```bash
ffmpeg -i /path/to/video.mp4 -f image2pipe -vcodec png pipe:1 | \
.venv/bin/waldo \
  --stdin-format png \
  --template /path/to/template.png \
  --output-csv tracks.csv
```

Track from an ffmpeg `image2pipe` JPEG pipeline:

```bash
ffmpeg -i /path/to/video.mp4 -f image2pipe -vcodec mjpeg pipe:1 | \
.venv/bin/waldo \
  --stdin-format jpeg \
  --template /path/to/template.png \
  --output-csv tracks.csv
```

Use stdin auto-detection with ffmpeg and let `waldo` infer PNG/JPEG streams automatically:

```bash
ffmpeg -i /path/to/video.mp4 -f image2pipe -vcodec png pipe:1 | \
.venv/bin/waldo \
  --template /path/to/template.png \
  --output-csv tracks.csv
```

If no explicit input source is provided and stdin is piped, `waldo` auto-switches to stdin mode. For raw stdin streams, frame size can come from either `--stdin-size WIDTHxHEIGHT` or `WALDO_STDIN_SIZE=WIDTHxHEIGHT`.

Version check:

```bash
.venv/bin/waldo --version
```

The CSV contains:

- `frame_index`
- `frame_id`
- `x,y,w,h`
- `confidence`
- `status` (`tracked`, `redetected`, `missing`)

## Example Artifacts

ROI template used for matching:

![ROI template](examples/roi_test/template.png)

Packaged sample video used for reproducible verification:

- [`examples/roi_test/videos/roi_test.mp4`](examples/roi_test/videos/roi_test.mp4)

Second input frame from the example set:

![Second input frame](examples/roi_test/frames/frame_001.png)

Second debug frame from the original frame-based run:

![Second debug frame](examples/roi_test/debug/000001.png)

Fourth input frame from the example set:

![Fourth input frame](examples/roi_test/frames/frame_003.png)

Fourth debug frame from the original frame-based run:

![Fourth debug frame](examples/roi_test/debug/000003.png)

Second debug frame from the ffmpeg stdin verification run:

![Second stdin debug frame](examples/roi_test/stdin_debug/000001.png)

Fourth debug frame from the ffmpeg stdin verification run:

![Fourth stdin debug frame](examples/roi_test/stdin_debug/000003.png)

Example CSV outputs:

- `examples/roi_test/tracks.csv`
- `examples/roi_test/stdin_tracks.csv`

Debug output directories:

- `examples/roi_test/debug/`
- `examples/roi_test/stdin_debug/`

## Features

- The tracker uses OpenCV normalized template matching with a local search window and periodic full-frame re-detection.
- It accepts ffmpeg pipeline input on stdin, including raw `bgr24` and concatenated PNG/JPEG `image2pipe` streams.
- It auto-detects piped stdin when no explicit input source is provided.
- It maintains both the original template and a slowly refreshed recent template so small text/content changes can be tolerated.
- If confidence falls below `--min-confidence`, the frame is marked `missing`.
- Omit `--debug-dir` or pass `--no-debug-images` to skip annotated image output entirely.
- Use `--debug-every N` to only save every Nth debug frame.

## Project Structure

```text
waldo/
├── AGENTS.md                    # contributor guide and repository workflow notes
├── IMPLEMENTATION_PLAN.md       # shared implementation plan and checklist
├── INSTALL                      # installation and release checklist
├── examples/                    # packaged verification assets
├── LICENSE                      # project MIT license
├── MANIFEST.in                  # explicit sdist allowlist
├── pep517_backend.py            # local PEP 517 backend shim for the filesystem rename workaround
├── project_structure            # local project structure reference file
├── pyproject.toml               # primary PEP 517 packaging metadata
├── README.md                    # user-facing project overview and usage guide
├── release_notes/
│   ├── RELEASE_NOTES_v1.0.0.md  # versioned release notes for v1.0.0
│   └── RELEASE_NOTES_v1.1.0.md  # versioned release notes for v1.1.0
├── requirements-dev.txt         # development and packaging tool dependencies
├── requirements.txt             # runtime dependencies
├── scripts/
│   └── install_local_pep517.sh* # local helper for the documented wheel-build install flow
├── setup.py                     # compatibility shim for older tooling
├── waldo/
│   ├── cli.py                   # packaged CLI and tracking implementation
│   ├── __init__.py              # package metadata, including version
│   └── __main__.py              # python -m waldo entrypoint
└── waldo.py                     # wrapper entrypoint
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Contributing

Contributions should follow the repository conventions documented in [`AGENTS.md`](AGENTS.md).

Before opening changes:

- update [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) when plans or implementation details change
- run the documented verification steps, at minimum `python -m py_compile` and `python -m waldo --help`
- keep release-facing docs such as [`README.md`](README.md) and [`INSTALL`](INSTALL) aligned with behavior changes

## Coding Style

- Use Python 3.10+ with 4-space indentation and `snake_case` naming for functions, variables, and modules.
- Keep the existing OpenCV-oriented structure and prefer small, focused functions over broad rewrites.
- Format Python changes with [Ruff](https://github.com/astral-sh/ruff) so the codebase stays consistent with the current style.

## Acknowledgements

- OpenAI Codex for coding-agent workflow support during development
- GPT-5.4 for implementation and documentation assistance

## Notes

- In this environment, `pip install .` can still fail after a successful PEP 517 wheel build because pip's own wheel-cache finalization hits `EXDEV`; the supported local workaround is `python -m build --no-isolation` followed by `pip install --no-deps dist/*.whl`.
- `pep517_backend.py` addresses the backend-side rename failure during `python -m build`; it does not patch pip's separate wheel-cache finalization step, which is why the direct wheel-install fallback is still documented.
- For raw stdin pipelines, `waldo` requires frame size from `--stdin-size` or `WALDO_STDIN_SIZE`; encoded PNG/JPEG stdin streams do not need an explicit size.

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

# Release Notes: v1.0.0

Tag: `v1.0.0`  
Release message: `Release 1.0.0`

## Summary

`v1.0.0` is the initial packaged release of `waldo`, a Python-based region-of-interest tracker for image frame sequences and video files.

## Highlights

- Introduced the core OpenCV tracker with hybrid local tracking, periodic re-detection, and conservative template refresh.
- Added CLI support for:
  - `--frames-dir`
  - `--video`
  - `--template`
  - `--init-bbox`
  - CSV output and annotated debug-frame generation
- Packaged the project as the `waldo` Python module with:
  - PEP 517-first packaging via `pyproject.toml`
  - a compatibility `setup.py`
  - a local `pep517_backend.py` workaround for `EXDEV` build-environment issues
- Added repository and release metadata:
  - `README.md`
  - `INSTALL`
  - `MANIFEST.in`
  - `requirements.txt`
  - `requirements-dev.txt`
  - `.gitignore`
  - `AGENTS.md`
- Included packaged example artifacts under `examples/roi_test/`, including:
  - template image
  - example frames
  - example debug frames
  - sample tracking CSV

## Operational Notes

- The documented local install flow builds a wheel first, then installs it directly to avoid environment-specific `pip install .` failures caused by `EXDEV`.
- Debug output can be enabled to inspect per-frame ROI tracking visually.

## Included Commit

- `2cd8b5c` `waldo - image region of interest tracker`

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

# Waldo Implementation Plan

## Summary

Implement a Python CLI tool that tracks a region of interest across either a frame directory or a video file. The tracker should support initialization from either a template image or a first-frame bounding box, then emit per-frame ROI coordinates, confidence, and status to CSV, with optional annotated debug frames.

## Todo Checklist

- [x] Implement the tracker CLI and packaged entrypoint as `waldo`.
- [x] Document installation, packaging, and release workflows.
- [x] Add release-manifest and repository hygiene controls.
- [x] Verify packaged execution, version reporting, and local install workflows.
- [x] Add a contributor guide in `AGENTS.md` for repository conventions and workflows.
- [x] Add phase-2 stdin pipeline input support for ffmpeg streaming.
- [x] Support raw `bgr24` stdin frames with size from `--stdin-size` or `WALDO_STDIN_SIZE`.
- [x] Support encoded `image2pipe` PNG/JPEG stdin streams.
- [x] Document ffmpeg pipeline usage and validate stdin tracking paths.
- [x] Create a reproducible sample video from `examples/roi_test/frames`.
- [x] Generate corresponding debug frames under `examples/roi_test/stdin_debug`.
- [x] Refresh README to document stdin pipeline support and packaged verification artifacts.
- [x] Review INSTALL for any remaining stdin-pipeline documentation gaps.
- [ ] Apply the MIT license header to all pertinent tracked text files.
- [ ] Preserve shebang placement and exclude binary assets plus machine-readable CSV artifacts from header insertion.
- [x] Add `License`, `Contributing`, and `Acknowledgements` sections to `README.md`.
- [x] Refresh `AGENTS.md` so it reflects the current project functionality and workflows.
- [x] Add coding style guidance and the Ruff reference to `README.md`.
- [x] Bump the project version to `1.1.0` across metadata and documentation.
- [x] Create the release commit and annotated tag `v1.1.0`.
- [x] Generate Markdown release notes for `v1.0.0` and `v1.1.0`.
- [x] Move release notes into `release_notes/` and add a README directory reference.
- [x] Rewrite the README project layout section as a directory tree with inline descriptions.
- [x] Rename the README section heading from `Project Layout` to `Project Structure`.

## Implementation Changes

- Add a single executable wrapper, `waldo.py`, with a CLI accepting:
  `--frames-dir` or `--video`
  `--template` or `--init-bbox x,y,w,h`
  optional `--output-csv`, `--debug-dir`, `--start-frame`, `--end-frame`, `--search-margin`, `--redetect-interval`, `--min-confidence`, `--scales`, `--recent-template-weight`, `--template-refresh-rate`
- Normalize both input modes into one frame iterator interface that yields `frame_index`, stable frame identifier, and RGB frame array.
- Implement a stateful hybrid tracker with:
  initialization from a provided template or from a cropped first-frame ROI
  local search around the previous ROI using normalized cross-correlation
  periodic or confidence-triggered full-frame re-detection
  conservative template refresh using both the original template and a recent accepted crop
- Emit CSV rows with:
  `frame_index, frame_id, x, y, w, h, confidence, status`
  where `status` is `tracked`, `redetected`, or `missing`
- Optionally write annotated debug images showing the detected box and confidence/status overlay.
- Optimize runtime for long frame sequences by caching grayscale multi-scale templates, narrowing local scale search around the last accepted match, and avoiding unnecessary periodic full-frame re-detections when confidence remains high.
- Optimize debug output generation by writing annotated PNGs asynchronously so disk I/O does not block tracking throughput.
- Add debug output controls so operators can skip most debug frames with `--debug-every N` or disable debug image generation entirely.
- Package the tracker as a distributable Python module with `requirements.txt`, install metadata, a console entrypoint, and explicit semantic versioning.
- Keep the package on PEP 517 and add a backend-level filesystem workaround for environments where setuptools artifact finalization hits `EXDEV` during `os.rename`.
- Add an operator-facing local install workaround for this environment: build via PEP 517 first, then install the produced wheel directly with `--no-deps` to avoid pip's wheel-cache rename failure.
- Keep a root `setup.py` as a compatibility shim for older setuptools workflows, with `pyproject.toml` remaining the primary metadata source for modern tooling.
- Document runtime requirements and example commands in `README.md`.
- Add a dedicated `INSTALL` document that describes the preferred PEP 517 installation path and the `setup.py` compatibility fallback.
- Add repository hygiene and release-boundary controls through `.gitignore` and an explicit `MANIFEST.in` allowlist so local artifacts stay out of git and out of release tarballs unless required.
- Add a third input mode that auto-detects piped `stdin` when no explicit input source is set, so ffmpeg can stream frames directly into `waldo`.
- Add stdin-specific options:
  `--stdin-format auto|raw-bgr24|png|jpeg`
  `--stdin-size WIDTHxHEIGHT`
- Support ffmpeg `rawvideo` pipelines using `bgr24`, with frame size resolved in this order:
  `--stdin-size`
  `WALDO_STDIN_SIZE`
  otherwise fail with a clear error.
- Support ffmpeg `image2pipe` pipelines for concatenated PNG and JPEG frames, including auto-detection of PNG/JPEG signatures from the initial stdin buffer.
- Keep all existing tracking, CSV, debug, and initialization options valid for stdin mode except `--frames-dir`.
- Generate timestamp-based `frame_id` values for stdin-fed frames.
- Add reproducible verification artifacts by packaging a small sample video built from the example frame set and the corresponding debug output produced by `waldo`.
- Apply the repository MIT license header consistently across source files, scripts, configuration files, and human-readable documentation using syntax-appropriate comment styles.

## Public Interfaces

- CLI entrypoint:
  `waldo [input] [initialization] [options]`
- Internal tracker API should expose:
  `initialize(frame, template | bbox) -> detection`
  `update(frame) -> detection`
- Detection objects should carry bounding box, confidence, and status, and config should be grouped in a dedicated dataclass for tuning.
- Package metadata should expose version `1.1.0`, and the `waldo.py` wrapper should remain aligned with the packaged CLI.
- Default generated artifact names should use the `waldo` project name where they represent project outputs rather than tracker-domain concepts.
- The PEP 517 backend should remain setuptools-based, but may be wrapped locally to replace failing cross-device renames with copy-and-replace behavior during wheel/sdist finalization.
- `setup.py` should be compatibility-only: modern setuptools should defer to `pyproject.toml`, while older setuptools should fall back to explicit metadata and package discovery.
- Installation documentation should distinguish runtime dependencies from development/build tooling and make the supported local install path explicit.
- Release maintenance should be driven by explicit package metadata and manifest policy rather than by whatever happens to be present in the git working tree.
- CLI input handling should accept auto-detected piped stdin as an alternative to `--video` and `--frames-dir`, with `--stdin-format` and `--stdin-size` controlling stdin decoding when needed.

## Test Plan

- Verify CLI parsing and startup for both `--frames-dir` and `--video`.
- Verify initialization from both `--template` and `--init-bbox`.
- Confirm stable tracking when the ROI shifts gradually and its text/content changes slightly.
- Confirm recovery by forcing local tracking failure and checking that full-frame re-detection resumes tracking.
- Confirm `missing` output when confidence stays below threshold.
- Verify CSV output shape and optional debug frame generation.
- Verify `--debug-every N` writes only the expected subset of annotated frames.
- Verify `--no-debug-images` suppresses debug output even when a debug directory is supplied.
- Verify packaged execution via `python -m waldo`, installed console script `waldo`, and `--version`.
- Verify default CLI outputs such as the CSV filename use the `waldo` project name.
- Verify stdin auto-detection activates only when stdin is piped and no explicit input source is selected.
- Verify raw `bgr24` stdin tracking with `--stdin-size`.
- Verify raw `bgr24` stdin tracking with `WALDO_STDIN_SIZE` and no `--stdin-size`.
- Verify missing raw stdin dimensions fail clearly.
- Verify PNG and JPEG `image2pipe` stdin streams decode and track correctly.
- Verify malformed or truncated stdin data fails clearly.
- Verify stdin CSV output uses timestamp-based `frame_id` values.
- Verify the license header is present in all intended tracked text files, shebang lines remain first where required, and excluded binary/CSV artifacts remain untouched.
- Verify `python -m build` and `pip install .` succeed under PEP 517 in the current environment after applying the filesystem workaround.
- Verify the fallback local install path succeeds: `python -m build --no-isolation` followed by `pip install --no-deps dist/*.whl`.
- Verify `python setup.py --version` still works for older-tool compatibility without conflicting with modern pyproject-driven metadata.
- Verify the dedicated `INSTALL` file matches the current PEP 517-first and `setup.py`-fallback workflow.
- Verify `.gitignore` covers generated directories and local environments such as `build/`, `tmp/`, `testing/`, `dist/`, `prev_versions/`, `.venv/`, `__pycache__/`, and `*.egg-info/`.
- Verify `MANIFEST.in` includes only release-required files, including `INSTALL`, and excludes local-only directories from sdists.
- Compare runtime on representative frame subsets and the full sample sequence, with and without debug image output.
- Run at minimum:
  `python3 -m py_compile waldo.py`
  `python3 -m py_compile waldo/cli.py`
  `python3 -m waldo --help`

## Assumptions

- The ROI mainly undergoes translation with small scale variation; major rotation or perspective distortion is out of scope for v1.
- ROI appearance changes are modest and remain correlated with the original template.
- Color information should be preserved during matching.
- Automatic recovery is preferred over aborting on tracking loss.
- Video decoding relies on `ffmpeg`/`ffprobe`; frame-directory mode relies on image files readable by Pillow.
- OpenCV is the intended runtime dependency, and debug image generation should favor throughput over maximum PNG compression.
- Version `1.1.0` is the current packaged release baseline.
- Some environments may report same-device paths to user-space while still causing backend artifact renames to fail with `EXDEV`; the workaround should treat that as an installation-environment quirk, not a package metadata error.
- `pip install .` may still fail in this environment after a successful wheel build if pip's own ephemeral wheel cache hits the same rename problem; direct installation of the built wheel is the expected fallback.
- Backward compatibility with older setup.py-based tooling is desirable, but `pyproject.toml` remains the source of truth for modern packaging metadata.
- The repository is intended to be maintained as a source tree with explicit release curation, not as a “ship everything in the repo” archive.
- Stdin pipeline support is limited to raw `bgr24` and ffmpeg-style concatenated PNG/JPEG `image2pipe` streams for phase 2.

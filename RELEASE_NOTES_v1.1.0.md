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

# Release Notes: v1.1.0

Tag: `v1.1.0`  
Release message: `Release 1.1.0`

## Summary

`v1.1.0` expands `waldo` with ffmpeg-friendly stdin pipeline support, refreshed documentation and contributor guidance, repository-wide MIT licensing metadata, and a packaged verification video plus stdin-debug artifacts.

## Highlights

- Added stdin pipeline input support so `waldo` can track ROIs from ffmpeg streams when no explicit input source is provided.
- Added stdin decoding modes for:
  - raw `bgr24`
  - PNG `image2pipe`
  - JPEG `image2pipe`
  - auto-detection of encoded stdin streams
- Added raw stdin size handling through:
  - `--stdin-size WIDTHxHEIGHT`
  - `WALDO_STDIN_SIZE=WIDTHxHEIGHT`
- Added timestamp-based `frame_id` values for stdin-fed frames.
- Added reproducible packaged verification artifacts:
  - `examples/roi_test/videos/roi_test.mp4`
  - `examples/roi_test/stdin_debug/`
  - `examples/roi_test/stdin_tracks.csv`
- Added repository-wide MIT license headers to tracked text/code/configuration files and added the root `LICENSE`.
- Updated contributor and user documentation:
  - refreshed `README.md`
  - refreshed `AGENTS.md`
  - renamed `IMPLEMENTATION_PLAN.doc` to `IMPLEMENTATION_PLAN.md`
  - added README sections for license, contributing, acknowledgements, and coding style
- Applied Ruff formatting to the Python CLI implementation.

## Upgrade Notes

- The project version is now `1.1.0`.
- Users relying on packaged wheel names or documented install commands should use `waldo-1.1.0-py3-none-any.whl`.
- stdin tracking remains limited to raw `bgr24` and ffmpeg-style concatenated PNG/JPEG streams in this release.

## Included Commits Since v1.0.0

- `c5e9283` `Add ffmpeg stdin pipeline support`
- `f7aefdd` `Add MIT license headers and LICENSE`
- `df4d013` `Update contributor guide`
- `8fee920` `Update README sections`
- `b02b6f1` `Update implementation plan`
- `19afe46` `Format CLI with Ruff`
- `e08518f` `Document coding style guidelines`
- `714a62f` `Bump version to 1.1.0`

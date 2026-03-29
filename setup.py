# waldo - image region of interest tracker
# Copyright (C) 2026 notweerdmonk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path

from setuptools import find_packages, setup
from setuptools import __version__ as setuptools_version


ROOT = Path(__file__).parent
VERSION_NS = {}
exec((ROOT / "waldo" / "__init__.py").read_text(), VERSION_NS)


def _major_version(raw: str) -> int:
    return int(raw.split(".", 1)[0])


if _major_version(setuptools_version) >= 61:
    # Modern setuptools can read metadata from pyproject.toml directly.
    setup()
else:
    # Fallback for older tooling that still relies on setup.py metadata.
    setup(
        name="waldo",
        version=VERSION_NS["__version__"],
        description="Track a moving region of interest across frames or video.",
        packages=find_packages(include=["waldo", "waldo.*"]),
        install_requires=["numpy", "opencv-python"],
        entry_points={"console_scripts": ["waldo=waldo.cli:run"]},
    )

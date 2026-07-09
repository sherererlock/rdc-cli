#!/usr/bin/env python3
"""Build a portable, self-contained rdc-cli distribution for Windows.

Assembles a directory that users can run without installing Python, pip,
or renderdoc.  Requires only Python 3.10+ stdlib to run.

Usage::

    python scripts/build_portable.py [--output DIR] [--python-version 3.14.4]

Prerequisites:
    - Compiled renderdoc artifacts at %LOCALAPPDATA%/rdc/renderdoc/
      (run ``pixi run setup-renderdoc`` first)
    - Android APK at %LOCALAPPDATA%/rdc/share/renderdoc/plugins/android/
      (run ``pixi run setup-renderdoc`` with ``--android`` first)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_PYTHON_VERSION = "3.14.4"
_PYTHON_EMBED_URL = (
    "https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
)
_ADB_URL = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"

_RENDERDOC_SRC = Path(os.environ.get("LOCALAPPDATA", "")) / "rdc"
_RENDERDOC_ARTIFACTS = ["renderdoc.pyd", "renderdoc.dll", "renderdoccmd.exe", "renderdoc.json"]
_RENDERDOC_REQUIRED = ["renderdoc.pyd", "renderdoc.dll"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing progress."""
    _log(f"  downloading {url}")
    urlretrieve(url, str(dest))


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip to *dest* with zip-slip protection."""
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(dest_resolved):
            sys.stderr.write(f"ERROR: zip-slip attempt: {member.filename}\n")
            raise SystemExit(1)
        zf.extract(member, dest)


# ---------------------------------------------------------------------------
# Step 1: Python embeddable
# ---------------------------------------------------------------------------


def step_download_python(out: Path, version: str) -> Path:
    """Download and extract Python embeddable package."""
    python_dir = out / "python"
    if python_dir.exists():
        _log("[1/6] Python embeddable already present, skipping")
        return python_dir

    _log(f"[1/6] Downloading Python {version} embeddable ...")
    url = _PYTHON_EMBED_URL.format(version=version)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        _download(url, tmp)
        python_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp) as zf:
            _safe_extractall(zf, python_dir)
    finally:
        tmp.unlink(missing_ok=True)

    # Patch ._pth to enable site-packages (uncomment "import site")
    major_minor = version.split(".")[:2]
    pth_name = f"python{''.join(major_minor)}._pth"
    pth_file = python_dir / pth_name
    if pth_file.exists():
        content = pth_file.read_text(encoding="utf-8")
        content = content.replace("#import site", "import site")
        pth_file.write_text(content, encoding="utf-8")
        _log(f"  patched {pth_name} to enable site-packages")
    else:
        _log(f"  WARNING: {pth_name} not found, site-packages may not work")

    return python_dir


# ---------------------------------------------------------------------------
# Step 2: Install rdc-cli + dependencies
# ---------------------------------------------------------------------------


def step_install_rdc(python_dir: Path) -> None:
    """Install rdc-cli from local source into the embeddable Python's site-packages."""
    site_packages = python_dir / "Lib" / "site-packages"

    # Check if already installed
    python_exe = python_dir / "python.exe"
    result = subprocess.run(
        [str(python_exe), "-m", "rdc", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _log("[2/6] rdc-cli already installed, skipping")
        return

    _log("[2/6] Installing rdc-cli + dependencies ...")
    site_packages.mkdir(parents=True, exist_ok=True)

    uv = shutil.which("uv")
    if uv:
        cmd = [
            uv, "pip", "install",
            str(_PROJECT_ROOT),
            "--target", str(site_packages),
            "--python", str(python_exe),
        ]
    else:
        pip = shutil.which("pip")
        if not pip:
            sys.stderr.write("ERROR: neither uv nor pip found in PATH\n")
            raise SystemExit(1)
        cmd = [
            pip, "install",
            str(_PROJECT_ROOT),
            "--target", str(site_packages),
            "--no-warn-script-location",
        ]

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 4: Copy renderdoc artifacts
# ---------------------------------------------------------------------------


def step_copy_renderdoc(out: Path, renderdoc_src: Path) -> None:
    """Copy compiled renderdoc files into the portable directory."""
    src = renderdoc_src / "renderdoc"
    dest = out / "renderdoc"

    if dest.exists() and all((dest / f).exists() for f in _RENDERDOC_REQUIRED):
        _log("[3/6] renderdoc artifacts already present, skipping")
        return

    _log("[3/6] Copying renderdoc artifacts ...")
    # Verify source exists
    for name in _RENDERDOC_REQUIRED:
        if not (src / name).exists():
            sys.stderr.write(f"ERROR: {src / name} not found\n")
            sys.stderr.write("Run 'pixi run setup-renderdoc' first.\n")
            raise SystemExit(1)

    dest.mkdir(parents=True, exist_ok=True)
    for name in _RENDERDOC_ARTIFACTS:
        artifact = src / name
        if artifact.exists():
            shutil.copy2(artifact, dest / name)
            _log(f"  copied {name}")
        else:
            _log(f"  skipped {name} (not found)")


# ---------------------------------------------------------------------------
# Step 5: Copy Android APK
# ---------------------------------------------------------------------------


def step_copy_apk(out: Path, renderdoc_src: Path) -> None:
    """Copy Android APK into the portable directory."""
    src = renderdoc_src / "share" / "renderdoc" / "plugins" / "android"
    dest = out / "share" / "renderdoc" / "plugins" / "android"

    if dest.exists() and list(dest.glob("*.apk")):
        _log("[4/6] Android APK already present, skipping")
        return

    _log("[4/6] Copying Android APK ...")
    if not src.exists() or not list(src.glob("*.apk")):
        _log("  WARNING: no APK found at %s, skipping" % src)
        _log("  Android capture will not work without the APK.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    for apk in src.glob("*.apk"):
        shutil.copy2(apk, dest / apk.name)
        _log(f"  copied {apk.name}")


# ---------------------------------------------------------------------------
# Step 6: Download ADB platform-tools
# ---------------------------------------------------------------------------


def step_download_adb(out: Path) -> None:
    """Download Android platform-tools (adb)."""
    dest = out / "platform-tools"
    adb = dest / "adb.exe"

    if adb.exists():
        _log("[5/6] ADB already present, skipping")
        return

    _log("[5/6] Downloading ADB platform-tools ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        _download(_ADB_URL, tmp)
        # Extract: the zip contains a top-level platform-tools/ directory
        with zipfile.ZipFile(tmp) as zf:
            _safe_extractall(zf, out)
    finally:
        tmp.unlink(missing_ok=True)

    if not adb.exists():
        sys.stderr.write(f"ERROR: adb.exe not found at {adb} after extraction\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Step 7: Generate bat files
# ---------------------------------------------------------------------------

_RDC_BAT = r"""@echo off
setlocal
set "DIR=%~dp0"
set "RENDERDOC_PYTHON_PATH=%DIR%renderdoc"
set "PATH=%DIR%platform-tools;%DIR%python;%PATH%"
"%DIR%python\python.exe" -c "from rdc.cli import entry; entry()" %*
endlocal
"""

_RDC_SHELL_BAT = r"""@echo off
set "DIR=%~dp0"
set "RENDERDOC_PYTHON_PATH=%DIR%renderdoc"
set "PATH=%DIR%platform-tools;%DIR%python;%PATH%"
doskey rdc="%DIR%python\python.exe" -c "from rdc.cli import entry; entry()" $*
echo.
echo  rdc-cli portable shell
echo  Type "rdc --help" to get started.
echo  Type "exit" to quit.
echo.
cmd /k "title rdc-cli"
"""


def step_generate_bat(out: Path) -> None:
    """Write launcher bat files."""
    _log("[6/6] Generating launcher scripts ...")

    (out / "rdc.bat").write_text(_RDC_BAT.lstrip(), encoding="utf-8")
    _log("  wrote rdc.bat")

    (out / "rdc-shell.bat").write_text(_RDC_SHELL_BAT.lstrip(), encoding="utf-8")
    _log("  wrote rdc-shell.bat")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable rdc-cli distribution for Windows.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "dist" / "rdc-portable",
        help="Output directory (default: dist/rdc-portable)",
    )
    parser.add_argument(
        "--python-version",
        default=_PYTHON_VERSION,
        help=f"Python embeddable version to download (default: {_PYTHON_VERSION})",
    )
    parser.add_argument(
        "--renderdoc-dir",
        type=Path,
        default=_RENDERDOC_SRC,
        help="Path to compiled renderdoc artifacts (default: %%LOCALAPPDATA%%/rdc)",
    )
    parser.add_argument(
        "--skip-adb",
        action="store_true",
        help="Skip downloading ADB platform-tools",
    )
    args = parser.parse_args(argv)

    out: Path = args.output.resolve()
    _log(f"=== Building portable rdc-cli at {out} ===")
    out.mkdir(parents=True, exist_ok=True)

    python_dir = step_download_python(out, args.python_version)
    step_install_rdc(python_dir)
    step_copy_renderdoc(out, args.renderdoc_dir)
    step_copy_apk(out, args.renderdoc_dir)
    if not args.skip_adb:
        step_download_adb(out)
    step_generate_bat(out)

    _log("")
    _log("=== Done ===")
    _log(f"Portable distribution at: {out}")
    _log("")
    _log("Quick test:")
    _log(f'  cd "{out}"')
    _log("  rdc.bat doctor")
    _log("  rdc-shell.bat")


if __name__ == "__main__":
    main()

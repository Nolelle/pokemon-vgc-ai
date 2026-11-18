"""Recreate the local Rust source distribution from the verified PyPI source and patch."""

import argparse
import gzip
import hashlib
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA256 = "a1d4be6eec408e8ec1d8540a85178d8efddd0a207865278f5c5d3c119e1722c0"
VERSION = "0.1.0+regmc.3"


def change_once(path, old, new):
    content = path.read_text()
    if content.count(old) != 1:
        raise ValueError(f"unexpected version metadata in {path}")
    path.write_text(content.replace(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_sdist", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.upstream_sdist.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        parser.error("expected the original pikapro-0.1.0 source archive from PyPI")
    directory = ROOT / "patches/pikapro"
    patches = sorted(directory.glob("*.patch"))
    if not patches:
        parser.error("missing Rust patch")
    with tempfile.TemporaryDirectory() as temp:
        with tarfile.open(args.upstream_sdist) as archive:
            archive.extractall(temp, filter="data")
        source = Path(temp) / "pikapro-0.1.0"
        for patch in patches:
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        change_once(source / "PKG-INFO", "\nVersion: 0.1.0\n", f"\nVersion: {VERSION}\n")
        change_once(
            source / "crates/pikapro-py/Cargo.toml",
            "version.workspace = true\nedition",
            f'version = "{VERSION}"\nedition',
        )
        change_once(
            source / "Cargo.lock",
            'name = "pikapro-py"\nversion = "0.1.0"',
            f'name = "pikapro-py"\nversion = "{VERSION}"',
        )
        name = f"pikapro-{VERSION}"
        output = directory / f"{name}.tar.gz"

        def normalized(info):
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            return info

        with (
            output.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        ):
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as archive:
                archive.add(source, arcname=name, filter=normalized)
        print(output)


if __name__ == "__main__":
    main()

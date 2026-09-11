#!/usr/bin/env python3
# This file is part of SNN2Bitstream.
# Copyright (C) 2026 Xindan Zhang, Sorbonne Université, CNRS, LIP6

# SNN2Bitstream is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SNN2Bitstream is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Download and prepare datasets under <ROOT>/data/.

MNIST via torchvision; N-MNIST, CIFAR10-DVS and DVS Gesture via tonic.

Usage:
    python tools/download_data.py {mnist|nmnist|cifar10dvs|dvsgesture|all}
"""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_mnist():
    """Download MNIST using torchvision.datasets.MNIST."""
    try:
        from torchvision import datasets
    except ImportError:
        print("[MNIST] torchvision is not installed. Please `pip install torchvision`.")
        return

    target_dir = os.path.join(ROOT, "data")
    print(f"[MNIST] Using root: {target_dir}")

    # This will create: data/MNIST/raw/...
    datasets.MNIST(target_dir, train=True, download=True)
    datasets.MNIST(target_dir, train=False, download=True)

    print("[MNIST] Download complete (data/MNIST/raw/...).")


def _nmnist_exists(root: str) -> bool:
    """Return True if N-MNIST seems to be already present under root."""
    nmnist_root = os.path.join(root, "NMNIST")
    train_dir = os.path.join(nmnist_root, "Train")
    test_dir = os.path.join(nmnist_root, "Test")
    return os.path.isdir(train_dir) and os.path.isdir(test_dir)


def ensure_nmnist():
    """Download N-MNIST using tonic.datasets.NMNIST."""
    try:
        import tonic
    except ImportError:
        print("[NMNIST] tonic is not installed. Please `pip install tonic`.")
        return

    # All scripts in this repo expect data/NMNIST/... structure.
    save_root = os.path.join(ROOT, "data")
    nmnist_root = os.path.join(save_root, "NMNIST")
    os.makedirs(nmnist_root, exist_ok=True)

    if _nmnist_exists(save_root):
        print(f"[NMNIST] Already present under {nmnist_root}, skip download.")
        return

    print(f"[NMNIST] Using save_root: {save_root}")
    print("[NMNIST] This may be ~250MB, please wait...")

    # tonic renamed the path kwarg: newer uses save_to, older uses data_dir.
    def _download_with_param(param_name: str):
        kwargs = {param_name: save_root}
        tonic.datasets.NMNIST(train=True,  **kwargs)
        tonic.datasets.NMNIST(train=False, **kwargs)

    try:
        _download_with_param("save_to")
    except TypeError:
        _download_with_param("data_dir")

    if _nmnist_exists(save_root):
        print(f"[NMNIST] Download complete (data/NMNIST/Train, data/NMNIST/Test).")
    else:
        print("[NMNIST] Download finished but data structure was not found.")
        print("         Please check the content of data/NMNIST manually.")


def _prefetch_cifar10dvs(archive: str, url: str, md5: str) -> bool:
    """Fetch the CIFAR10-DVS archive ourselves, resuming a partial file.

    tonic points at ``figshare.com/ndownloader/...``, which answers with an empty
    HTTP 202 ("accepted, not ready") that torchvision reports as a corrupt file.
    The ``ndownloader.figshare.com`` host serves the same object directly, so we
    put the archive in place and let tonic move straight on to extraction.
    """
    import hashlib
    import urllib.request

    def digest(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(archive) and digest(archive) == md5:
        print("[CIFAR10DVS] Archive already present.")
        return True

    os.makedirs(os.path.dirname(archive), exist_ok=True)
    part = archive + ".part"
    for attempt in range(1, 6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
                total = resp.length + have if resp.length else None
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    if total:
                        print(f"\r[CIFAR10DVS] {have / 2**30:.2f} / {total / 2**30:.2f} GiB",
                              end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[CIFAR10DVS] attempt {attempt} interrupted ({e}); resuming.")
            continue
        if digest(part) == md5:
            os.replace(part, archive)
            return True
        print(f"[CIFAR10DVS] attempt {attempt}: checksum mismatch, restarting.")
        os.remove(part)
    print("[CIFAR10DVS] Could not fetch the archive.")
    return False


def ensure_cifar10dvs():
    """Download CIFAR10-DVS using tonic.datasets.CIFAR10DVS."""
    try:
        import tonic
    except ImportError:
        print("[CIFAR10DVS] tonic is not installed. Please `pip install tonic`.")
        return

    save_root = os.path.join(ROOT, "data")
    print(f"[CIFAR10DVS] Using save_root: {save_root}")
    print("[CIFAR10DVS] The archive is about 10 GiB, please wait...")

    cls = tonic.datasets.CIFAR10DVS
    if not _prefetch_cifar10dvs(
        os.path.join(save_root, "CIFAR10DVS", cls.filename),
        "https://ndownloader.figshare.com/files/38023437",
        cls.file_md5,
    ):
        return

    try:
        tonic.datasets.CIFAR10DVS(save_to=save_root)
        print("[CIFAR10DVS] Download complete.")
    except TypeError:
        try:
            tonic.datasets.CIFAR10DVS(data_dir=save_root)
            print("[CIFAR10DVS] Download complete.")
        except Exception as e:
            print(f"[CIFAR10DVS] Download failed: {e}")


def ensure_dvsgesture():
    """Download IBM DVS Gesture dataset via tonic."""
    try:
        import tonic
    except ImportError:
        print("[DVSGesture] tonic is not installed. Please `pip install tonic`.")
        return

    save_root = os.path.join(ROOT, "data")
    print(f"[DVSGesture] Using save_root: {save_root}")
    print("[DVSGesture] This may be ~1.6GB, please wait...")

    try:
        tonic.datasets.DVSGesture(save_to=save_root, train=True)
        tonic.datasets.DVSGesture(save_to=save_root, train=False)
        print("[DVSGesture] Download complete.")
    except TypeError:
        try:
            tonic.datasets.DVSGesture(data_dir=save_root, train=True)
            tonic.datasets.DVSGesture(data_dir=save_root, train=False)
            print("[DVSGesture] Download complete.")
        except Exception as e:
            print(f"[DVSGesture] Download failed: {e}")


def usage():
    print(
        "Usage:\n"
        "  python tools/download_data.py mnist       # MNIST\n"
        "  python tools/download_data.py nmnist      # N-MNIST\n"
        "  python tools/download_data.py cifar10dvs  # CIFAR10-DVS\n"
        "  python tools/download_data.py dvsgesture  # IBM DVS Gesture\n"
        "  python tools/download_data.py all         # all datasets\n"
    )


def main():
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "mnist":
        ensure_mnist()
    elif cmd == "nmnist":
        ensure_nmnist()
    elif cmd == "cifar10dvs":
        ensure_cifar10dvs()
    elif cmd == "dvsgesture":
        ensure_dvsgesture()
    elif cmd == "all":
        ensure_mnist()
        ensure_nmnist()
        ensure_cifar10dvs()
        ensure_dvsgesture()
    else:
        usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

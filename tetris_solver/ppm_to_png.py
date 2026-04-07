#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_token(fp) -> bytes:
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PPM header")
        line = line.strip()
        if not line or line.startswith(b"#"):
            continue
        return line


def read_ppm_p6(path: Path) -> np.ndarray:
    with path.open("rb") as fp:
        magic = fp.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Unsupported PPM format: {magic!r}, expected P6")

        dims = _read_token(fp).split()
        if len(dims) != 2:
            raise ValueError("Invalid dimensions in PPM header")
        width, height = int(dims[0]), int(dims[1])

        max_value = int(_read_token(fp))
        if max_value != 255:
            raise ValueError(f"Unsupported max value: {max_value}, expected 255")

        raw = fp.read(width * height * 3)
        if len(raw) != width * height * 3:
            raise ValueError("Pixel data size does not match dimensions")

    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return arr


def main() -> None:
    input_path = Path("output/final_board.ppm")
    output_path = input_path.with_suffix(".png")

    image = read_ppm_p6(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)

    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Thin entry point that applies GPU visibility before importing model code."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _requested_gpus(arguments: list[str]) -> str:
    for index, argument in enumerate(arguments):
        if argument == "--gpus" and index + 1 < len(arguments):
            return arguments[index + 1].strip().casefold()
        if argument.startswith("--gpus="):
            return argument.partition("=")[2].strip().casefold()
    return "auto"


gpus = _requested_gpus(sys.argv[1:])
if gpus == "none":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif gpus != "auto":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# Keep direct execution working even when the package has not been installed
# in editable mode yet.
source_root = Path(__file__).resolve().parents[1]
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

main = importlib.import_module("oasis.mock_experiments").main


if __name__ == "__main__":
    raise SystemExit(main())

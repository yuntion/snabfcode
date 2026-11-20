"""Public MLIR import and HEIR-to-Lattigo compilation for the reference arm."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from heir.heir_cli.heir_cli import HeirOptBackend, HeirTranslateBackend


def import_mlir(path: Path) -> str:
    """Load linalg/secret MLIR; HEIR parses and verifies the returned module."""
    if path.suffix != ".mlir":
        raise ValueError(f"expected .mlir input, got {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class LattigoSource:
    input_mlir: str
    lowered_mlir: str
    go_source: str
    diagnostics: str


def compile_reference(
    model: Path,
    *,
    profile: Path,
    min_slots: int,
    heir_opt: Path,
    heir_translate: Path,
    package: str = "baseline",
    importer: Callable[[Path], str] = import_mlir,
) -> LattigoSource:
    """Compile the fixed-profile control arm through the native HEIR pipeline.

    Project importers may produce the same linalg/secret MLIR contract from
    another model format, including domain annotations, without forking HEIR.
    """
    if min_slots < 1 or min_slots & (min_slots - 1):
        raise ValueError("min_slots must be a positive power of two")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", package) or package == "_":
        raise ValueError(f"invalid Go package identifier: {package}")
    profile = profile.resolve(strict=True)
    if any(char in str(profile) for char in '"\n\r\\'):
        raise ValueError("unsupported character in HEIR parameter file path")
    module = importer(model)
    lowered, opt_diagnostics = HeirOptBackend(heir_opt).run_binary_stderr(
        [
            "--annotate-module=backend=lattigo scheme=ckks",
            "--linalg-canonicalizations",
            f'--torch-linalg-to-ckks=min-slot-count={min_slots} parameter-file="{profile}"',
            "--scheme-to-lattigo",
        ],
        module,
    )
    generated, translate_diagnostics = HeirTranslateBackend(
        heir_translate
    ).run_binary_stderr(["--emit-lattigo", f"--package-name={package}"], lowered)
    return LattigoSource(
        module, lowered, generated, opt_diagnostics + translate_diagnostics
    )

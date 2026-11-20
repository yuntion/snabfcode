"""Command line entry for generating the fixed-profile CKKS reference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from heir.heir_cli.heir_cli import CLIError

from .compiler import compile_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--min-slots", required=True, type=int)
    parser.add_argument("--heir-opt", required=True, type=Path)
    parser.add_argument("--heir-translate", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = compile_reference(
            args.model,
            profile=args.profile,
            min_slots=args.min_slots,
            heir_opt=args.heir_opt,
            heir_translate=args.heir_translate,
        )
    except CLIError as error:
        print(f"HEIR failed: {error.command}\n{error.stderr}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "input.mlir").write_text(result.input_mlir, encoding="utf-8")
    (args.out_dir / "lowered.mlir").write_text(result.lowered_mlir, encoding="utf-8")
    (args.out_dir / "model.go").write_text(result.go_source, encoding="utf-8")
    (args.out_dir / "heir-diagnostics.log").write_text(
        result.diagnostics, encoding="utf-8"
    )
    manifest = {
        "arm": "heir_fixed_profile_reference",
        "input_sha256": hashlib.sha256(result.input_mlir.encode("utf-8")).hexdigest(),
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "heir_opt": str(args.heir_opt.resolve()),
        "heir_translate": str(args.heir_translate.resolve()),
        "min_slots": args.min_slots,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

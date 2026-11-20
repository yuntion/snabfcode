#!/usr/bin/env bash
# Re-run the checked compiler gates against immutable inputs after Pass 1 cleanup.
set -euo pipefail
root=/home/fhecompiler
cd "$root"
opt=$(realpath "$1")
mkdir -m 700 "$2"
out=$(realpath "$2")
exec > >(tee "$out/run.log") 2>&1
r="$root/build/pass4-ab-release-20261119/run.POAcXfhz"
export LD_LIBRARY_PATH="$root/extern/mlir24/usr/lib/llvm-24/lib:$root/extern/mlir24/usr/lib/x86_64-linux-gnu:$root/extern/highs-install/lib"
unset PYTHONOPTIMIZE PYTHONPATH SNABF_PASS4_ESTIMATES SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS GOFLAGS
export PYTHONDONTWRITEBYTECODE=1
python3 -c 'import json,hashlib,sys; from pathlib import Path; r=Path(sys.argv[1]); m=json.loads(Path("doc/pass4-ab-results-2026-11-19/sha256.json").read_text())["members"]; keys=[f"{c}/{n}" for c in ("reuse64","cross_residual","residual") for n in ("input.mlir","cache.json")]; assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==m[k] for k in keys); print("6/6 frozen inputs matched")' "$r"
sha256sum "$opt" "$root/extern/heir-release/heir/heir-translate" > "$out/tools.sha256"
export HEIR_TRANSLATE="$root/extern/heir-release/heir/heir-translate"
bash src/compiler/Target/Lattigo/regenerate-heir-helpers.sh --check
export PATH="$root/extern/go/bin:$PATH" GOPATH="$root/extern/gopath" GOCACHE="$root/extern/go-cache" GOMAXPROCS=3 GOTOOLCHAIN=local GOPROXY=off
go test -mod=readonly -count=1 ./src/backend/ringswitch/epochs/...
python3 src/compiler/pass4/tests/replay_gates.py "$opt" "$r/reuse64/input.mlir" "$r/reuse64/cache.json" "$out/replay" --crossing "$r/cross_residual/input.mlir" --cross-estimates "$r/cross_residual/cache.json" --abi-shell "$r/residual/input.mlir" --shell-estimates "$r/residual/cache.json"
export SNABF_PASS4_ESTIMATES="$r/reuse64/cache.json" SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS=128
python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" "$r/reuse64/input.mlir" | tee "$out/direct.json"
python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" "$r/reuse64/input.mlir" --pass4 | tee "$out/mixed.json"
cmp "$r/gate-direct.json" "$out/direct.json"
cmp "$r/gate-mixed.json" "$out/mixed.json"
printf 'PASS: native Pass 4 gates; both comparison records unchanged\n'

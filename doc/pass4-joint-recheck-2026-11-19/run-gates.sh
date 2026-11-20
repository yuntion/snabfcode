#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); run=$(realpath "$2"); d=$(realpath "$3")
cd "$root"
ext="$root/extern"; opt="$d/native/ir/snabf-opt"; here=$(cd "$(dirname "$0")" && pwd -P)
unset PYTHONOPTIMIZE PYTHONPATH HEIR_BIN_DIR LD_PRELOAD GOFLAGS GODEBUG SNABF_PASS4_ESTIMATES SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PATH="$ext/go/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ext/mlir24/usr/lib/llvm-24/lib:$ext/mlir24/usr/lib/x86_64-linux-gnu:$ext/highs-install/lib"
export HEIR_TRANSLATE="$ext/heir-release/heir/heir-translate" LATTICE_ESTIMATOR="$ext/lattice-estimator"
export GOPATH="$ext/gopath" GOMODCACHE="$ext/gopath/pkg/mod" GOCACHE="$d/go-cache" GOTOOLCHAIN=local GOPROXY=off GOWORK=off GOMAXPROCS=4 TMPDIR="$d/test-tmp"
mkdir -m 700 "$TMPDIR"
python3 "$here/check-inputs.py" "$root" "$run" "$d" "$here/B-source.sha256" > "$d/inputs.json"
python3 src/compiler/pass4/tests/replay_gates.py "$opt" "$run/reuse64/input.mlir" "$run/reuse64/cache.json" "$d/replay" --crossing "$run/cross_residual/input.mlir" --cross-estimates "$run/cross_residual/cache.json" --abi-shell "$run/residual/input.mlir" --shell-estimates "$run/residual/cache.json" | tee "$d/b-replay.log"
bash src/compiler/Target/Lattigo/regenerate-heir-helpers.sh --check
sha256sum src/backend/ringswitch/epochs/heirckks/standard.go | tee "$d/heir-helper.sha256"
go test -mod=readonly -count=1 -v ./src/backend/ringswitch/epochs/... | tee "$d/go-test.log"
export SNABF_PASS4_ESTIMATES="$run/reuse64/cache.json" SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS=128
python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" "$run/reuse64/input.mlir" | tee "$d/direct.json"
python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" "$run/reuse64/input.mlir" --pass4 | tee "$d/mixed.json"
cmp "$d/direct.json" "$run/gate-direct.json"
cmp "$d/mixed.json" "$run/gate-mixed.json"

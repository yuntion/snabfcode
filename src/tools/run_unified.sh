#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
sdk=${SNABF_MLIR_ROOT:-$root/extern/mlir24/usr/lib/llvm-24}
heir=${HEIR_BIN_DIR:-$root/extern/heir-release/heir}
jobs=${JOBS:-5}
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${GO_BIN_DIR:-$root/extern/go/bin}:$PATH" GOPATH="$root/extern/gopath" GOCACHE="$root/extern/go-cache" GOTOOLCHAIN=local GOMAXPROCS="$jobs"
export PYTHONPATH="$root/src/frontend:$root/extern/heir/frontend${PYTHONPATH:+:$PYTHONPATH}"
out=build/unified
mkdir -p "$out"
cmake -S src/compiler -B build/native -DCMAKE_BUILD_TYPE=Release -DSNABF_MLIR_ROOT="$sdk" -DCMAKE_PREFIX_PATH="$root/extern/highs-install"
cmake --build build/native --parallel "$jobs"
ctest --test-dir build/native --output-on-failure
go build -mod=readonly -o "$out/goemit" ./src/compiler/Target/Lattigo/go-bind
python3 -m snabf_frontend.unified --root "$root" --out "$out" --snabf-opt build/native/ir/snabf-opt --heir-bin "$heir" --goemit "$out/goemit"
python3 test/compiler/test_unified_ir.py "$root" | tee "$out/ir-tests.log"
# Execute the identical acceptance harness; only change its generated package/fixture root.
sed 's|build/pass1/|build/unified/|g' test/pass1/inference_test.go > "$out/acceptance.go.txt"
python3 -c 'import json,os; from pathlib import Path; print(json.dumps({"Replace": {str(Path("test/pass1/inference_test.go").resolve()): str(Path("build/unified/acceptance.go.txt").resolve())}}))' > "$out/overlay.json"
go test -mod=readonly -count=1 -overlay "$out/overlay.json" -v ./test/pass1 ./test/unified | tee "$out/inference.log"

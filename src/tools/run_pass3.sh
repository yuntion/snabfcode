#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
sdk=${SNABF_MLIR_ROOT:-$root/extern/mlir24/usr/lib/llvm-24}
heir=${HEIR_BIN_DIR:-$root/extern/heir-release/heir}
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$root/extern/go/bin:$PATH" GOPATH="$root/extern/gopath" GOCACHE="$root/extern/go-cache" GOPROXY=off GOTOOLCHAIN=local GOMAXPROCS=${JOBS:-4}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src/frontend:$root/extern/heir/frontend${PYTHONPATH:+:$PYTHONPATH}"
export LATTICE_ESTIMATOR=${LATTICE_ESTIMATOR:-$root/extern/lattice-estimator}
test -d "$LATTICE_ESTIMATOR/estimator" || { echo 'Set LATTICE_ESTIMATOR to the pinned checkout (see doc/pass3-results)'; exit 2; }
mkdir -p build/pass3
out=$(mktemp -d "$root/build/pass3/run.XXXXXXXX")
exec > >(tee "$out/run.log") 2>&1
echo "Pass 3 experiment: $out"
cmake -S src/compiler -B build/pass3/native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSNABF_MLIR_ROOT="$sdk" -DCMAKE_PREFIX_PATH="$root/extern/highs-install"
cmake --build build/pass3/native -j"${JOBS:-4}"
ctest --test-dir build/pass3/native --output-on-failure
go test -mod=readonly -count=1 -v ./src/backend/gadget ./test/backend
go build -mod=readonly -o "$out/goemit" ./src/compiler/Target/Lattigo/go-bind
python3 -m snabf_frontend.pass3 --root "$root" --out "$out" --heir-bin "$heir" --snabf-opt "$root/build/pass3/native/ir/snabf-opt" --goemit "$out/goemit"
go test -mod=readonly -count=1 -timeout 60m -v "./${out#"$root"/}/go_test"
python3 src/tools/check_pass3.py "$out" "$root/build/pass3/native/ir/snabf-opt"
python3 src/tools/collect_pass3.py "$out" "$out/report"

#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
sdk=${SNABF_MLIR_ROOT:-$root/extern/mlir24/usr/lib/llvm-24}
heir=${HEIR_BIN_DIR:-$root/extern/heir-release/heir}
export PATH="${GO_BIN_DIR:-$root/extern/go/bin}:$PATH" LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GOPATH="$root/extern/gopath" GOCACHE="$root/extern/go-cache" GOTOOLCHAIN=local GOPROXY=off GOMAXPROCS=${JOBS:-5}
export PYTHONPATH="$root/src/frontend:$root/extern/heir/frontend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
export LATTICE_ESTIMATOR=${LATTICE_ESTIMATOR:-$root/extern/lattice-estimator}
test -d "$LATTICE_ESTIMATOR/estimator" || { echo 'Set LATTICE_ESTIMATOR to the pinned checkout (doc/pass3-development.md)' >&2; exit 2; }
base=$(realpath -m "${SNABF_PASS1_BUILD:-$root/build/pass1-revision}")
case "$base/" in "$root/build/"?*/) ;; *) printf 'Pass 1 build must be under %s/build: %s\n' "$root" "$base" >&2; exit 2;; esac
mkdir -p "$base"; out=$(mktemp -d "$base/run.XXXXXXXX")
native="$base/native"; opt="$native/ir/snabf-opt"
exec > >(tee "$out/run.log") 2>&1
echo "Pass 1 evidence: $out"
cmake -S src/compiler -B "$native" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSNABF_MLIR_ROOT="$sdk" -DHEIR="$root/extern/heir" -DCMAKE_PREFIX_PATH="$root/extern/highs-install"
cmake --build "$native" -j"$GOMAXPROCS"
ctest --test-dir "$native" --output-on-failure --no-tests=error
go test -mod=readonly -count=1 -v ./test/backend
go build -mod=readonly -o "$out/goemit" ./src/compiler/Target/Lattigo/go-bind
flags=(--root "$root" --heir-bin "$heir" --snabf-opt "$opt" --goemit "$out/goemit")
python3 -m snabf_frontend.unified "${flags[@]}" --out "$out"
python3 test/compiler/test_unified_ir.py "$root" "$out" "$opt"
bash src/tools/pass1_boot.sh "$out/periodic_boot" "$opt" "$heir" "$out/goemit"
python3 src/tools/pass1_evidence.py overlay "$out"
go test -mod=readonly -overlay "$out/overlay.json" -count=1 -p=1 -timeout=30m -v ./test/pass1 ./test/unified | tee "$out/inference.log"
mkdir "$out/pass2" "$out/pass3"
python3 -m snabf_frontend.pass1_feedback "${flags[@]}" --out "$out/pass2"
go test -mod=readonly -count=1 -v "./${out#"$root"/}/pass2/go_test" | tee "$out/pass2/inference.log"
python3 -m snabf_frontend.pass3 "${flags[@]}" --out "$out/pass3"
go test -mod=readonly -count=1 -timeout=60m -v "./${out#"$root"/}/pass3/go_test" | tee "$out/pass3/inference.log"
python3 src/tools/check_pass3.py "$out/pass3" "$opt"
python3 src/tools/pass1_evidence.py collect "$out"

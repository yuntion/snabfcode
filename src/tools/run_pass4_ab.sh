#!/usr/bin/env bash
# Real source graphs, independently checked plans, fresh ciphertexts and actual keys.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
ext=${SNABF_EXTERN:-$root/extern}
sdk=${SNABF_MLIR_ROOT:-$ext/mlir24/usr/lib/llvm-24}
heir=${HEIR_BIN_DIR:-$ext/heir-release/heir}
export LD_LIBRARY_PATH="$sdk/lib:$sdk/../x86_64-linux-gnu:$ext/highs-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ext/go/bin:$PATH" GOPATH="$ext/gopath" GOCACHE="${GOCACHE:-$ext/go-cache}" GOPROXY=off GOTOOLCHAIN=local
export PYTHONPATH="$root/src/frontend:$ext/heir/frontend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export LATTICE_ESTIMATOR=${LATTICE_ESTIMATOR:-$ext/lattice-estimator} GOMAXPROCS=${JOBS:-4}
work=$(realpath -m "${SNABF_AB_BUILD:-$root/build/pass4-ab}")
# Paired test packages import the two generated models via this Go module.
case "$work/" in "$root/"?*/) ;; *) printf 'SNABF_AB_BUILD must be inside the project Go module: %s\n' "$root" >&2; exit 2;; esac
mkdir -p "$work"
out=$(mktemp -d "$work/run.XXXXXXXX"); out=$(realpath "$out")
exec > >(tee "$out/run.log") 2>&1
printf 'Pass 4 A/B evidence: %s\n' "$out"
cmake -S src/compiler -B "$work/native" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSNABF_MLIR_ROOT="$sdk" -DHEIR="$ext/heir" -DCMAKE_PREFIX_PATH="$ext/highs-install"
cmake --build "$work/native" -j"${JOBS:-4}"
ctest --test-dir "$work/native" --output-on-failure
bash src/compiler/Target/Lattigo/regenerate-heir-helpers.sh --check
go test -mod=readonly -count=1 ./test/backend ./src/backend/gadget ./src/backend/ringswitch/epochs/...
go build -mod=readonly -o "$out/keyshape" ./src/tools/pass4_keyshape
go build -mod=readonly -o "$out/go-bind" ./src/compiler/Target/Lattigo/go-bind
python3 -m snabf_frontend.pass4_ab --root "$root" --out "$out" --heir-bin "$heir" --snabf-opt "$work/native/ir/snabf-opt" --goemit "$out/go-bind" control strided64 reuse64 residual cross_residual
python3 test/pass4/verify_ab.py "$root" "$out" "$work/native/ir/snabf-opt" "$out/keyshape"
(find src/compiler src/backend src/frontend src/tools test -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.mlir' -o -name '*.td' -o -name '*.go' -o -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
echo "Verified: $out"

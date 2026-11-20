#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
jobs=${JOBS:-8}
heir_source=${HEIR_SOURCE_DIR:-$root/extern/heir}
heir_bin=${HEIR_BIN_DIR:-$root/extern/heir-release/heir}
go_bin=${GO_BIN_DIR:-$root/extern/go/bin}
highs=${HIGHS_PREFIX:-$root/extern/highs-install}

export PATH="$go_bin:$PATH"
export GOPATH=${GOPATH:-$root/extern/gopath}
export GOCACHE=${GOCACHE:-$root/extern/go-cache}
export GOTOOLCHAIN=local
export GOMAXPROCS=${GOMAXPROCS:-$jobs}
export PYTHONPATH="$root/src/frontend:$heir_source/frontend${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p build/logs
HEIR_BIN_DIR="$heir_bin" python3 -m unittest discover -s test/frontend -v \
  2>&1 | tee build/logs/frontend-tests.log
python3 -m snabf_frontend test/fixtures/linear_silu_linear.mlir \
  --profile test/fixtures/reference_profile.json --min-slots 4096 \
  --heir-opt "$heir_bin/heir-opt" --heir-translate "$heir_bin/heir-translate" \
  --out-dir build/baseline
gofmt -w build/baseline/model.go

cmake -S src/compiler -B build/compiler -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$highs"
cmake --build build/compiler --parallel "$jobs"
ctest --test-dir build/compiler --output-on-failure | tee build/logs/compiler-tests.log
go mod verify
go test -mod=readonly -p "$jobs" -count=1 -v ./test/backend ./test/ckks \
  2>&1 | tee build/logs/reference-tests.log

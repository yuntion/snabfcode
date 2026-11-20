#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
sdk=${SNABF_MLIR_ROOT:-$root/extern/mlir24/usr/lib/llvm-24}
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p build/pass2
run=$(mktemp -d "$root/build/pass2/accept.XXXXXXXX")
cmake -S src/compiler -B build/native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DSNABF_MLIR_ROOT="$sdk" -DCMAKE_PREFIX_PATH="$root/extern/highs-install" | tee "$run/configure.log"
cmake --build build/native --parallel "${JOBS:-5}" | tee "$run/build.log"
ctest --test-dir build/native --output-on-failure | tee "$run/ctest.log"
export SNABF_P2_OUT="$run/experiment" SNABF_OPT="$root/build/native/ir/snabf-opt"
bash src/tools/run_pass2.sh
python3 src/tools/collect_pass2.py "$SNABF_P2_OUT" "${1:-$run/report}" --ctest "$run/ctest.log"

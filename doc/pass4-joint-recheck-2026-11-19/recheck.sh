#!/usr/bin/env bash
set -euo pipefail
root=/home/fhecompiler
run="$root/build/pass4-ab-release-20261119/run.POAcXfhz"
here=$(cd "$(dirname "$0")" && pwd -P)
cd "$root"
d=$(mktemp -d "$root/build/pass4-joint-recheck.XXXXXXXX")
printf 'fresh run: %s\n' "$d"
sdk="$root/extern/mlir24/usr/lib/llvm-24"
export LD_LIBRARY_PATH="$sdk/lib:$root/extern/mlir24/usr/lib/x86_64-linux-gnu:$root/extern/highs-install/lib"
cmake -S src/compiler -B "$d/native" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSNABF_MLIR_ROOT="$sdk" -DHEIR="$root/extern/heir" -DCMAKE_PREFIX_PATH="$root/extern/highs-install" 2>&1 | tee "$d/configure.log"
cmake --build "$d/native" -j6 2>&1 | tee "$d/build.log"
ctest --test-dir "$d/native" --output-on-failure 2>&1 | tee "$d/ctest.log"
bash "$here/run-gates.sh" "$root" "$run" "$d" 2>&1 | tee "$d/gates.log"
env -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 python3 "$here/check-extra-final.py" "$root" "$run" "$d" 2>&1 | tee "$d/extra.log"
env -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 SNABF_PASS1_CLI="$d/native/snabf-pass1" PYTHONPATH="$root/src/compiler/pass1" python3 -m unittest discover -v -s test/compiler -p test_pass1.py 2>&1 | tee "$d/pass1-legacy-test.log"

#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$root"
ext=${SNABF_EXTERN:-$root/extern}
sdk=${SNABF_MLIR_ROOT:-$ext/mlir24/usr/lib/llvm-24}
export LD_LIBRARY_PATH="$sdk/lib:$sdk/../x86_64-linux-gnu:$ext/highs-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ext/go/bin:$PATH" GOPATH="$ext/gopath" GOCACHE="${GOCACHE:-$ext/go-cache}" GOPROXY=off GOTOOLCHAIN=local GOMAXPROCS=${JOBS:-4} PYTHONDONTWRITEBYTECODE=1
export LATTICE_ESTIMATOR=${LATTICE_ESTIMATOR:-$ext/lattice-estimator}
work=${SNABF_PASS4_BUILD:-$root/build/pass4};mkdir -p "$work"
out=$(mktemp -d "$work/run.XXXXXXXX");exec > >(tee "$out/run.log") 2>&1;echo "Pass 4 experiment: $out"
cmake -S src/compiler -B "$work/native" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSNABF_MLIR_ROOT="$sdk" -DHEIR="$ext/heir" -DCMAKE_PREFIX_PATH="$ext/highs-install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
cmake --build "$work/native" -j"${JOBS:-4}";ctest --test-dir "$work/native" --output-on-failure
opt="$work/native/ir/snabf-opt";go test -count=1 ./test/backend ./src/backend/gadget;go build -o "$out/keyshape" ./src/tools/pass4_keyshape
archive=${SNABF_PASS3_EVIDENCE:-$root/doc/pass3-results/evidence.tar.gz};sha256sum "$archive" > "$out/pass3-input.sha256"
for net in reuse64 residual; do
 d="$out/$net";mkdir -p "$d";tar xzf "$archive" -C "$out" "$net/joint/checked.mlir" "$net/joint/plan.json" "$net/corpus.json"
 "$out/keyshape" "$d/joint/plan.json" > "$d/key-sizes.json"
 slots=$(python3 -c 'import re,sys; s=open(sys.argv[1]).read();print(int(re.search(r"scheme.requested_slot_count = (\d+)",s)[1]))' "$d/joint/checked.mlir")
 sage -python src/tools/pass4_security.py "$d/joint/plan.json" "$d/key-sizes.json" "$d/security.json" "$LATTICE_ESTIMATOR" --slots "$slots"
 export SNABF_PASS4_ESTIMATES="$d/security.json"
 "$opt" --allow-unregistered-dialect --snabf-plan --snabf-check-plan "$d/joint/checked.mlir" -o "$d/current.mlir"
 for arm in baseline adaptive expensive; do
  a="$d/$arm";mkdir "$a";case $arm in baseline) opt4="min-ring=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["ring"])' "$d/joint/plan.json")";; adaptive) opt4="";; expensive) opt4="switch-price=1000";; esac
  "$opt" --allow-unregistered-dialect "--snabf-pass4=$opt4 audit=$a/search.json" "$d/current.mlir" -o "$a/planned.mlir"
  "$opt" --allow-unregistered-dialect --snabf-check-plan --snabf-lower-ckks "$a/planned.mlir" -o "$a/lowered.mlir"
  "$opt" --allow-unregistered-dialect "--snabf-bind-lattigo=ring-go=$a/model.go audit=$a/program.json" "$a/lowered.mlir" -o /dev/null
  cp test/pass4/encrypted_test.go "$a/encrypted_test.go";SNABF_EVIDENCE="$a" SNABF_CORPUS="$d/corpus.json" go test -count=1 -timeout 30m -v "$a/model.go" "$a/encrypted_test.go"
 done
 python3 src/tools/check_pass4.py "$d" "$opt"
done
(cd "$root" && find src/compiler src/backend src/tools test/pass4 -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.td' -o -name '*.go' -o -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
echo "Verified: $out"

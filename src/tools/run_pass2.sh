#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$root"
sdk=${SNABF_MLIR_ROOT:-$root/extern/mlir24/usr/lib/llvm-24}
heir=${HEIR_BIN_DIR:-$root/extern/heir-release/heir}
opt=${SNABF_OPT:-$root/build/native/ir/snabf-opt}
export PATH="${GO_BIN_DIR:-$root/extern/go/bin}:$PATH"
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$root/src/frontend:${HEIR_SOURCE_DIR:-$root/extern/heir}/frontend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off
export GOPATH=${GOPATH:-$root/extern/gopath} GOCACHE=${GOCACHE:-$root/extern/go-cache} GOMAXPROCS=${JOBS:-5}
for tool in "$opt" "$heir/heir-opt" "$heir/heir-translate"; do
    [[ -x $tool ]] || { printf 'Missing prebuilt tool: %s\n' "$tool" >&2; exit 2; }
done
# Explicit run names are single-use. The default always gets a fresh directory.
mkdir -p build/pass2
if [[ -n ${SNABF_P2_OUT:-} ]]; then
    out=$(realpath -m -- "$SNABF_P2_OUT")
    [[ $out == "$root/build/"* ]] || { echo 'SNABF_P2_OUT must be inside the module build/ directory' >&2; exit 2; }
    mkdir -- "$out"
else
    out=$(mktemp -d "$root/build/pass2/run.XXXXXXXX")
fi
printf 'Pass 2 evidence directory: %s\n' "$out"
exec > >(tee "$out/run.log") 2>&1
go build -mod=readonly -o "$out/goemit" ./src/compiler/Target/Lattigo/go-bind
python3 -m unittest discover -s test/pass2 -p 'test_*.py' -v
options=()
if [[ -n ${SNABF_P2_EXTRA_ROTATION_CAP:-} ]]; then options+=(--extra-rotation-cap "$SNABF_P2_EXTRA_ROTATION_CAP"); fi
python3 -m snabf_frontend.pass2 prepare --root "$root" --out "$out" --heir-bin "$heir" \
    --snabf-opt "$opt" --goemit "$out/goemit" "${options[@]}"
python3 test/compiler/test_pass2_ir.py --root "$root" --out "$out" --snabf-opt "$opt" --heir-bin "$heir"
# Numerical hard failures are deferred until each workload has written all of
# its trials. Always attempt independent export and retain either failure code.
go_status=0
go test -mod=readonly -count=1 -timeout=90m -v "./${out#"$root"/}/go_test" | tee "$out/inference.log" || go_status=$?
export_status=0
python3 -m snabf_frontend.pass2 export --out "$out" || export_status=$?
if (( ! export_status && ! go_status )); then
    SNABF_P2_TEST_RUN="$out" python3 -m unittest discover -s test/pass2 -p test_evidence.py -v
fi
if (( go_status || export_status )); then
    printf 'Pass 2 FAILED (encrypted tests=%d, evidence export=%d); artifacts retained: %s\n' "$go_status" "$export_status" "$out" >&2
    if (( go_status )); then exit "$go_status"; fi
    exit "$export_status"
fi
printf 'Regression gates passed; see per-arm numerical PASS/FAIL in %s/evidence.json\n' "$out"

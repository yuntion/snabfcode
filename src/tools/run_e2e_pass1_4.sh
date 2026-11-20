#!/usr/bin/env bash
# Direct C++ emission; the JSON program is an audit artifact, not an interpreter input.
set -euo pipefail
usage() {
  printf '%s\n' 'run_e2e_pass1_4.sh --mlir FILE --estimates CACHE [--from source|pass3]' \
    '  [--corpus JSON] [--profile JSON] [--security JSON] [--oracle-mlir SOURCE (for pass3)]' \
    '  [--strategy adaptive|baseline|expensive] [--sw-key-bits N] [--out-root DIR]' \
    '  [--scenario inference|reuse64-128|reuse64-512|cross-residual|equiv-single-ring]' \
    '  [--workload control|strided64|reuse64|residual (required for equiv-single-ring)] [--pass3-options JSON]' \
    'source: run native Pass 1-4 on the supplied dense Linear/SiLU/Linear graph at a locked profile.' \
    'pass3: verify and replay a concrete single-ring checkpoint, without claiming fresh P1-3 synthesis.' \
    'Corpus, profile and Pass 3 security report default to siblings of the frozen estimator cache.'
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
root=$(realpath -e "${SRC_ROOT:-$(dirname "$0")/../..}")
ext=$(realpath -m "${SNABF_EXTERN:-$root/extern}")
sdk=$(realpath -m "${SNABF_MLIR_ROOT:-$ext/mlir24/usr/lib/llvm-24}")
heir=$(realpath -m "${HEIR_BIN_DIR:-$ext/heir-release/heir}")
build=$(realpath -m "${BUILD_DIR:-${SNABF_PASS4_BUILD:-$root/build/e2e-native}}")
outputs=${OUT_ROOT:-$root/build/e2e-runs}
export HEIR_TRANSLATE=$(realpath -m "${HEIR_TRANSLATE:-$heir/heir-translate}")
mlir= oracle= corpus= profile= security= estimates=${SNABF_PASS4_ESTIMATES:-}
mode=source strategy= env_bits=${SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS:-128} bits= jobs=${JOBS:-4}
scenario=inference workload= search_opts=
while (($#)); do
  case "$1" in -h|--help) usage; exit 0;; esac
  (($# >= 2)) || die "missing value for $1"
  case "$1" in
    --mlir) mlir=$2;; --from) mode=$2;; --oracle-mlir) oracle=$2;; --estimates) estimates=$2;;
    --scenario) scenario=$2;; --workload) workload=$2;; --pass3-options) search_opts=$2;;
    --corpus) corpus=$2;; --profile) profile=$2;; --security) security=$2;;
    --strategy) strategy=$2;; --sw-key-bits) bits=$2;; --out-root) outputs=$2;; *) die "unknown option $1";;
  esac
  shift 2
done
case "$mode" in source|pass3) ;; *) die 'expected --from source or pass3';; esac
case "$scenario" in
  inference) [[ -z $workload ]] || die '--workload requires a named benchmark';;
  reuse64-128|reuse64-512|cross-residual)
    name=reuse64; floor=128
    if [[ $scenario == cross-residual ]]; then name=cross_residual; fi
    if [[ $scenario == reuse64-512 ]]; then floor=512; fi
    [[ ${workload:-$name} == "$name" && ${bits:-$floor} == "$floor" && ${strategy:-adaptive} == adaptive ]] || die 'conflicting named scenario, policy or workload'
    workload=$name; bits=$floor;;
  equiv-single-ring)
    case "$workload" in control|strided64|reuse64|residual) ;; *) die 'paired scenario requires a registered --workload';; esac
    [[ ${strategy:-baseline} == baseline ]] || die 'paired comparison requires the fixed source ring'
    strategy=baseline;;
  *) die 'unknown scenario';;
esac
strategy=${strategy:-adaptive}; bits=${bits:-128}
case "$strategy" in adaptive|baseline|expensive) ;; *) die 'unknown strategy';; esac
[[ $bits =~ ^[0-9]+$ && $env_bits =~ ^[0-9]+$ && $jobs =~ ^[1-9][0-9]*$ ]] || die 'invalid security floor or JOBS'
# An explicit option may strengthen the process policy, never weaken it. The
# native pass also takes the maximum with the floor stored on the input IR.
requested_bits=$bits
bits=$(python3 -c 'import sys; n=max(128,*map(int,sys.argv[1:])); sys.exit("security floor overflows i64") if n >= 2**63 else print(n)' "$bits" "$env_bits")
[[ -f $mlir && -f $estimates ]] || die 'supply --mlir and frozen --estimates'
if [[ $mode == source ]]; then
  [[ -z $oracle ]] || die '--from source takes its oracle from --mlir'
  oracle=$mlir
else
  [[ -f $oracle ]] || die '--from pass3 also requires the original --oracle-mlir'
fi
siblings=$(dirname "$estimates")
corpus=${corpus:-$siblings/corpus.json}; profile=${profile:-$siblings/profile.json}; security=${security:-$siblings/security.json}
for file in "$corpus" "$profile" "$security"; do [[ -f $file ]] || die "missing locked input $file"; done
for tool in "$HEIR_TRANSLATE" "$heir/heir-opt" "$ext/go/bin/go"; do [[ -x $tool ]] || die "missing tool $tool"; done
outputs=$(realpath -m "$outputs"); mkdir -p "$outputs"
run=$(mktemp -d "$outputs/run.XXXXXXXX"); mkdir "$run"/{inputs,work,out,logs}
exec > >(tee "$run/run.log") 2>&1
printf 'RUN_DIR=%s\n' "$run"
stage=inputs
trap 'r=$?; if ((r)); then printf "FAILED stage=%s status=%s run=%s\n" "$stage" "$r" "$run" >&2; fi' EXIT
cp -- "$oracle" "$run/inputs/source.mlir"
if [[ $mode == pass3 ]]; then cp -- "$mlir" "$run/inputs/checkpoint.mlir"; fi
if [[ -n $search_opts ]]; then cp -- "$search_opts" "$run/inputs/pass3-options.json"; fi
cp -- "$estimates" "$run/inputs/estimates.json"; cp -- "$corpus" "$run/inputs/corpus.json"
cp -- "$profile" "$run/inputs/profile.json"; cp -- "$security" "$run/inputs/security.json"
printf 'SRC_ROOT=%s\nBUILD_DIR=%s\nJOBS=%s\nHEIR_TRANSLATE=%s\nMLIR=%s\nMODE=%s\nSTRATEGY=%s\nSNABF_PASS4_SWITCHING_KEY_SECURITY_BITS=%s\nESTIMATES_SOURCE=%s\nSNABF_PASS4_ESTIMATES=%s\n' \
  "$root" "$build" "$jobs" "$HEIR_TRANSLATE" "$(realpath "$mlir")" "$mode" "$strategy" "$bits" "$(realpath "$estimates")" "$run/inputs/estimates.json" > "$run/env.snapshot"
printf 'CLI_SWITCH_BITS=%s\nENV_SWITCH_BITS=%s\n' "$requested_bits" "$env_bits" >> "$run/env.snapshot"
printf 'SCENARIO=%s\nWORKLOAD=%s\n' "$scenario" "$workload" >> "$run/env.snapshot"
export PATH="$ext/go/bin:$PATH" GOPATH="$ext/gopath" GOCACHE="${GOCACHE:-$ext/go-cache}" GOPROXY=off GOTOOLCHAIN=local GOWORK=off GOFLAGS= GOMAXPROCS="$jobs"
export LD_LIBRARY_PATH="$sdk/lib:$sdk/../x86_64-linux-gnu:$ext/highs-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$root/src/frontend:$ext/heir/frontend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export SNABF_PASS4_ESTIMATES="$run/inputs/estimates.json" SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS="$bits"
step() {
  stage=$1; shift
  printf '[%s]\n' "$stage"
  printf '%q ' "$@" >> "$run/logs/commands.shlog"; printf '\n' >> "$run/logs/commands.shlog"
  "$@" 2>&1 | tee "$run/logs/$stage.log"
}
cd "$run"
sha256sum inputs/* > inputs.sha256
if [[ -n $workload ]]; then
  expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$root/test/pass4/matrix_sources.json" "$workload")
  printf '%s  inputs/source.mlir\n' "$expected" | sha256sum -c -
fi
step configure cmake -S "$root/src/compiler" -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DSNABF_MLIR_ROOT="$sdk" -DHEIR="$ext/heir" -DCMAKE_PREFIX_PATH="$ext/highs-install"
step build cmake --build "$build" --target snabf-opt -j"$jobs"
opt="$build/ir/snabf-opt"; native=("$opt" --allow-unregistered-dialect --mlir-print-op-on-diagnostic=false --verify-each)
step heir_helpers bash "$root/src/compiler/Target/Lattigo/regenerate-heir-helpers.sh" --check
# File-list build/test is run in the real module even when the evidence is outside it.
cd "$root"
stage=toolchain
module=$(go list -mod=readonly -m -f '{{.Version}} {{.Dir}}{{if .Replace}} replaced{{end}}' github.com/tuneinsight/lattigo/v6)
[[ $module == "v6.2.0 $(realpath -e "$ext/lattigo")" ]] || die "Lattigo module not locked: $module"
printf 'GO=%s\nLATTIGO=%s\n' "$(go version)" "$module" >> "$run/env.snapshot"
sha256sum "$opt" "$heir/heir-opt" "$HEIR_TRANSLATE" "$ext/go/bin/go" go.mod go.sum \
  src/backend/ringswitch/epochs/heirckks/standard.go src/tools/run_e2e_pass1_4.sh \
  src/frontend/snabf_frontend/{e2e_source,e2e_equiv,pass2,pass3,pass2_evidence}.py src/tools/{verify_e2e_pass1_4,e2e_matrix_evidence}.py \
  test/pass4/{encrypted_test.go,paired_test.go,e2e_adapters.go.in,matrix_sources.json} > "$run/toolchain.sha256"
if [[ $mode == source || $scenario == equiv-single-ring ]]; then
  step key_meter go build -mod=readonly -o "$run/work/key-meter" ./src/compiler/Target/Lattigo/go-bind
fi
cd "$run"
if [[ $mode == source ]]; then
  step pass1_3 python3 -m snabf_frontend.e2e_source --root "$root" --heir-bin "$heir" --snabf-opt "$opt" --goemit "$run/work/key-meter"
else
  step replay_checkpoint "${native[@]}" --snabf-check-plan inputs/checkpoint.mlir -o work/checkpoint.mlir
fi
policy='audit=out/search.json'
case "$strategy" in
  baseline) ring=$(python3 -c 'import json; print(1 << json.load(open("inputs/profile.json"))["logN"])'); policy+=" min-ring=$ring";;
  expensive) policy+=' switch-price=1000';;
esac
step pass4 "${native[@]}" "--snabf-pass4=$policy" work/checkpoint.mlir -o work/planned.mlir
step check "${native[@]}" --snabf-check-plan work/planned.mlir -o work/checked.mlir
step lower "${native[@]}" --snabf-lower-ckks work/checked.mlir -o work/lowered.mlir
step emit "${native[@]}" '--snabf-bind-lattigo=ring-go=out/model.go audit=out/program.json' work/lowered.mlir -o /dev/null
cd "$root"
step go_build go build -mod=readonly -o "$run/work/model.a" "$run/out/model.go"
export SNABF_CORPUS="$run/inputs/corpus.json" SNABF_EVIDENCE="$run/out"
if [[ $scenario == equiv-single-ring ]]; then
  cd "$run"
  step legacy_emit python3 -m snabf_frontend.e2e_equiv --root "$root" --heir-bin "$heir" --snabf-opt "$opt" --goemit "$run/work/key-meter"
  # Standalone comparison module: no rewriting Go packages or requiring run to live under root.
  mkdir comparison
  cp -- "$root/go.mod" "$root/go.sum" .
  step pair_module go mod edit -module=snabf.local/e2ecase -require=snabf.local/compiler@v0.0.0 "-replace=snabf.local/compiler=$root"
  cp -- "$root/test/pass4/e2e_adapters.go.in" comparison/adapters.go
  cp -- "$root/test/pass4/paired_test.go" comparison/paired_test.go
  step pair_build go build -mod=readonly ./out ./work/legacy
  export SNABF_PROGRAM="$run/out/program.json"
  step encrypted go test -mod=readonly -tags=snabf_generated -count=1 -timeout=30m -run '^TestPairedCiphertexts$' -v ./comparison
else
  cp -- test/pass4/encrypted_test.go "$run/out/encrypted_test.go"
  step encrypted go test -mod=readonly -tags=snabf_generated -count=1 -timeout=30m -v "$run/out/model.go" "$run/out/encrypted_test.go"
fi
cd "$run"
step inputs_recheck sha256sum -c inputs.sha256
step independent python3 "$root/src/tools/verify_e2e_pass1_4.py" "$run" --output "$run/verified.json"
cp -- verified.json acceptance.json
printf 'DONE: %s\n' "$run"

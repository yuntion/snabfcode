#!/usr/bin/env bash
# Runtime slice only. No graph arm, security approval or cut-margin feedback.
# Usage: bash test/compiler/phase2_fullslot_linear_runtime.sh PASS1 NEW_OUT [OPT]
# Set PATH to the pinned Go toolchain and GOPATH to a populated offline cache.
# Optional OPT records the pre-integration Pass 4 refusal on the frozen graph.
set -euo pipefail
umask 077
[[ $# -ge 2 && $# -le 3 ]] || { echo "usage: $0 PASS1 NEW_OUT [OPT]" >&2; exit 2; }
pass1=$(realpath -- "$1")
out=$(realpath -m -- "$2")
opt=''
if [[ $# -eq 3 ]]; then opt=$(realpath -- "$3"); fi
[[ -x $pass1 && ( -z $opt || -x $opt ) ]]
mkdir -m 700 -- "$out" # never overwrite an earlier observation
cd "$(dirname -- "$0")/../.."
export SNABF_TEST_PASS1="$pass1" SNABF_A12_EVIDENCE="$out"
export GOWORK=off GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local GOTELEMETRY=off
go version > "$out/go-version.txt"
sha256sum "$pass1" "$(command -v go)" go.mod go.sum src/backend/tower/*.go \
  src/backend/ringswitch/{fullslot*,repack*}.go test/compiler/phase2_fullslot_linear_runtime.sh > "$out/inputs.sha256"
go test -mod=readonly -count=1 -v -run '^TestFullSlot' ./src/backend/tower ./src/backend/ringswitch \
  2>&1 | tee "$out/gate.log"
[[ -s $out/observations.json && -s $out/pass1-request.json && -s $out/pass1-answer.json ]]
if [[ -n $opt ]]; then
  fixture=doc/e2e-matrix-2026-11-20-r3/frozen/reuse64/checkpoint.mlir
  sha256sum "$opt" "$fixture" >> "$out/inputs.sha256"
  rc=0
  "$opt" --allow-unregistered-dialect --verify-each --mlir-print-op-on-diagnostic=false \
    '--snabf-pass4=cut-profile=full-slot-split-v1' "$fixture" -o "$out/should-not-emit.mlir" \
    > "$out/pass4-refusal.log" 2>&1 || rc=$?
  printf 'exit=%d\n' "$rc" >> "$out/pass4-refusal.log"
  [[ $rc -eq 1 && ! -s $out/should-not-emit.mlir ]]
  grep -q 'error: unsupported_cut_profile' "$out/pass4-refusal.log"
fi
(cd "$out" && sha256sum -- *.log *.json *.txt inputs.sha256 > SHA256SUMS)

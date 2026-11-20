#!/usr/bin/env bash
# Called from run_pass1.sh with its pinned environment and fresh output path.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
out=$1; opt=$2; heir=$3; emitter=$4
mkdir "$out"
flags=(--allow-unregistered-dialect --verify-each)
"$heir/heir-opt" --lattigo-configure-crypto-context --mlir-print-op-generic --mlir-print-local-scope \
  "$root/test/fixtures/pass1_periodic_bootstrap.mlir" -o "$out/imported.mlir"
"$opt" "${flags[@]}" --pass-pipeline='builtin.module(snabf-pass1-candidates,snabf-plan,snabf-check-plan)' "$out/imported.mlir" -o "$out/checked.mlir"
"$opt" "${flags[@]}" --snabf-lower-ckks "$out/checked.mlir" -o "$out/lowered.mlir"
"$opt" "${flags[@]}" --snabf-bind-lattigo="audit=$out/audit.json" "$out/lowered.mlir" -o "$out/bound.mlir"
python3 "$root/test/compiler/test_boot_binding.py" "$opt" "$out"
"$heir/heir-opt" "$out/bound.mlir" -o "$out/verified.mlir"
"$heir/heir-translate" --emit-lattigo --package-name=periodicboot "$out/verified.mlir" -o "$out/translated.go.txt"
"$emitter" emit-bootstrap "$out/audit.json" "$out/translated.go.txt" "$out/model.go"
cp "$root/test/unified/periodic_bootstrap_test.go.in" "$out/periodic_bootstrap_test.go"
cd "$root"; go test -mod=readonly -count=1 -timeout=20m -v "./${out#"$root"/}" | tee "$out/inference.log"

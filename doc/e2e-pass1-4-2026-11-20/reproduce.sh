#!/usr/bin/env bash
# Reproduce either paired experiment without using any historical build/run as input.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$here/../.." && pwd -P)
mode=${1:-source}
case "$mode" in source|pass3) ;; *) printf 'usage: %s [source|pass3]\n' "$0" >&2; exit 2;; esac
[[ ${SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS:-128} == 128 ]] || { echo 'This paired regression needs an environment floor of 128 or unset; it will not weaken it.' >&2; exit 2; }
(cd "$here" && sha256sum -c frozen.sha256)
inputs="$here/frozen"
base=$(realpath -m "${OUT_ROOT:-$root/build/e2e-reproduce}")
mkdir -p "$base"; dest=$(mktemp -d "$base/pair.XXXXXXXX")
printf 'PAIR_DIR=%s\n' "$dest"
args=(--from "$mode" --mlir "$inputs/source.mlir")
if [[ $mode == pass3 ]]; then args=(--from pass3 --mlir "$inputs/checkpoint.mlir" --oracle-mlir "$inputs/source.mlir"); fi
for bits in 128 512; do
  "$root/src/tools/run_e2e_pass1_4.sh" "${args[@]}" --strategy adaptive --estimates "$inputs/estimates.json" --sw-key-bits "$bits" --out-root "$dest/$bits"
done
positive=("$dest/128"/run.*); strict=("$dest/512"/run.*)
python3 "$root/src/tools/verify_e2e_pass1_4.py" "${positive[0]}" --reuse64-pair "${strict[0]}" --output "$dest/pair.json"
printf 'PAIRED: %s/pair.json\n' "$dest"

#!/usr/bin/env bash
# Install or check the unmodified output of the pinned upstream translator.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$here/../../../.." && pwd -P)
dest="$root/src/backend/ringswitch/epochs/heirckks"
tool=${HEIR_TRANSLATE:-${HEIR_BIN_DIR:-$root/extern/heir-release/heir}/heir-translate}
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'expected --check or --write' >&2; exit 2;; esac
mkdir -p "$dest"
tmp=$(mktemp "$dest/.standard.XXXXXXXX")
trap 'rm -f -- "$tmp"' EXIT
"$tool" --emit-lattigo --package-name=heirckks "$here/StandardCKKS.mlir" -o "$tmp"
if [[ $mode == --check ]]; then
  cmp -- "$tmp" "$dest/standard.go"
else
  chmod 0644 "$tmp"
  mv -- "$tmp" "$dest/standard.go"
fi

#!/usr/bin/env bash
# Project-local x86_64 Ubuntu 22.04 SDK; no system installation.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
dir="$root/extern/mlir24"
version='24~++20261117042646+59551e9f844e-1~exp1~20261117042658.3439'
url='https://apt.llvm.org/jammy/pool/main/l/llvm-toolchain-snapshot'
mkdir -p "$dir/debs"
while read -r pkg sum; do
  file="${pkg}_${version}_amd64.deb"
  if [[ ! -f "$dir/debs/$file" ]]; then
    curl -fLsS --retry 3 --connect-timeout 15 --max-time 360 "$url/$file" -o "$dir/debs/$file.part"
    printf '%s  %s\n' "$sum" "$dir/debs/$file.part" | sha256sum -c -
    mv "$dir/debs/$file.part" "$dir/debs/$file"
  fi
  printf '%s  %s\n' "$sum" "$dir/debs/$file" | sha256sum -c -
  dpkg-deb -x "$dir/debs/$file" "$dir"
done <<'PACKAGES'
libllvm24 4b0942759d22b1e827946b2ef2ab3b78e2d5d41aaa93c7538f8861f23ac84bf1
libmlir-24 562f4c28a2ad0549c4b00bc9aff6596086189edd1e617db51ee9fd38ec242f4f
llvm-24-dev d9ba7345166c9a9fc85ae883293c4fefff00316239fb41d66444a0a3f2ac06c6
libmlir-24-dev fe7c7f6c4dbe788b5e1cd7c464bae2b9a28f0509f8af84a106e2adb5a7f4f4e3
mlir-24-tools 84294c90319903ad0e8bbba64e405e0d8f67d4a4da74af2492f50330c1fb9ef1
PACKAGES
LD_LIBRARY_PATH="$dir/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$dir/usr/lib/llvm-24/bin/mlir-tblgen" --version

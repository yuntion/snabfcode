#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
jobs=${JOBS:-8}
mkdir -p "$root/build/logs" "$root/extern/bazel-distdir"
cd "${HEIR_SOURCE_DIR:-$root/extern/heir}"
bazel=${BAZEL_BIN:-$root/extern/bazel-bin/bazel}
if [[ $# == 0 ]]; then
  set -- //tests/Examples/lattigo/ckks/sigmoid:sigmoid_test \
         //tests/Examples/lattigo/ckks/matvec_batchnorm:matvec_batchnorm_test
fi

"$bazel" --output_user_root="$root/extern/bazel-cache" --batch test \
  --jobs="$jobs" --local_resources="cpu=$jobs" \
  --distdir="$root/extern/bazel-distdir" \
  --remote_cache= --bes_backend= --bes_results_url= \
  "$@" 2>&1 | tee "$root/build/logs/heir-source-test.log"

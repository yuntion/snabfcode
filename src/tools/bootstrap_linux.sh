#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  printf 'Use a Linux x86-64 host for the pinned prebuilt toolchain.\n' >&2
  exit 1
fi

check_sha() {
  printf '%s  %s\n' "$2" "$1" | sha256sum -c -
}

fetch() {
  local url=$1 path=$2 sha=$3
  if [[ ! -f "$path" ]]; then
    curl -fLsS --retry 3 --connect-timeout 15 --max-time 240 \
      --continue-at - "$url" -o "$path.part"
    check_sha "$path.part" "$sha"
    mv -- "$path.part" "$path"
  else
    check_sha "$path" "$sha"
  fi
}

mkdir -p extern/bazel-bin extern/heir-release
heir_commit=368d0b17ee5dfc26e5da2fb417c63e368cff8ab7
heir_archive="extern/heir-$heir_commit.tar.gz"
fetch "https://codeload.github.com/google/heir/tar.gz/$heir_commit" "$heir_archive" \
  9fa7c572f76c660b36edae5cbc5f32147fd8b3d1c6a7fb391764c8607d7a19c7
if [[ ! -d extern/heir ]]; then
  mkdir -p extern/heir
  tar -xzf "$heir_archive" --strip-components=1 -C extern/heir
fi

wheel=heir_py-2026.11.1-cp310-abi3-manylinux_2_28_x86_64.whl
fetch "https://files.pythonhosted.org/packages/f1/9f/44fb103a4bafb97081e72240297d2c9034b831640dddc83dcf66572fa7a1/$wheel" \
  "extern/heir-release/$wheel" 2c76999bdf890c4132b3b603cd7cb7039c94775c543c3a986d9defd6fe0c266a
unzip -oq "extern/heir-release/$wheel" -d extern/heir-release heir/heir-opt heir/heir-translate

fetch https://releases.bazel.build/8.4.0/release/bazel-8.4.0-linux-x86_64 \
  extern/bazel-bin/bazel 43445203739a7fd4eb28abbbee8803f24998e6259cc618a3323fb32bd7fa4f30
chmod +x extern/bazel-bin/bazel

fetch https://dl.google.com/go/go1.26.0.linux-amd64.tar.gz \
  extern/go1.26.0.linux-amd64.tar.gz aac1b08a0fb0c4e0a7c1555beb7b59180b05dfc5a3d62e40e9de90cd42f88235
if [[ ! -x extern/go/bin/go ]]; then
  tar -xzf extern/go1.26.0.linux-amd64.tar.gz -C extern
fi
bash src/tools/prefetch_go_deps.sh

fetch https://codeload.github.com/ERGO-Code/HiGHS/tar.gz/refs/tags/v1.16.0 \
  extern/highs-v1.16.0.tar.gz 0fa9da512955fb08c6cca7bfb274ca12e8a0e8a12e44189bd6e0c355ce7701bc
if [[ ! -d extern/highs ]]; then
  mkdir -p extern/highs
  tar -xzf extern/highs-v1.16.0.tar.gz --strip-components=1 -C extern/highs
fi
if [[ ! -f extern/highs-install/lib/cmake/highs/highs-config.cmake ]]; then
  cmake -S extern/highs -B extern/highs-build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$root/extern/highs-install"
  cmake --build extern/highs-build --parallel "${JOBS:-8}"
  cmake --install extern/highs-build
fi

extern/bazel-bin/bazel --version
extern/heir-release/heir/heir-opt --version
extern/go/bin/go version
extern/highs-install/bin/highs --version

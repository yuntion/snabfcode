# Pinned development dependencies

Use local installations under `extern/`. No `sudo` or global installation is
required. On Linux x86-64, with a C++20 compiler, CMake, zlib development
headers, Python 3.10+, curl, tar, and unzip:

```bash
bash src/tools/bootstrap_linux.sh
```

| Dependency | Pin | Purpose |
|---|---|---|
| Google HEIR source | `368d0b17ee5dfc26e5da2fb417c63e368cff8ab7` | Python frontend and native extension development |
| Official `heir-py` Linux wheel | `2026.11.1` | Released `heir-opt` and `heir-translate` for the fixed control arm |
| Bazel | `8.4.0` | Version required by the pinned HEIR source |
| HiGHS | `v1.16.0` | C++ MILP SDK; installed into `extern/highs-install` |
| Go | `1.26.0` | Lattigo backend toolchain |
| Lattigo | `v6.2.0` | Native CKKS runtime. Browse `extern/lattigo/` for `core/rlwe`, `schemes/ckks`, `circuits/ckks` and examples. |

`bootstrap_linux.sh` also runs `bash src/tools/prefetch_go_deps.sh`. This fetches
the complete module graph from the project Go lockfiles into `extern/gopath`,
verifies it, and makes `extern/lattigo` a relative link to the same read-only
module source used for builds. No `go get`, `replace`, or second checkout is
needed. To verify an existing cache without downloading anything:

```bash
GOPROXY=off bash src/tools/prefetch_go_deps.sh
```

The project Windows mirror has a browsable snapshot extracted from the same
verified Go module zip. Neither source view is an editable fork.

The bootstrap script checks archive SHA-256 digests. There are deliberately
two HEIR pins: one for the released CLI binaries and one for the source tree.
Do not report a test of the former as a build of the latter. HEIR's own
upstream Go tests use its independently pinned Lattigo version.

On the experiment server, `sum.golang.org` over its default URL timed out.
To fetch through the reachable mirror with the same Go signing key, use:

```bash
export GOSUMDB='sum.golang.org https://sum.golang.google.cn'
```

If the default module proxy is also unreachable, use `GOPROXY=https://goproxy.cn`
for the fetch; the versions and required checksums do not change.

The Go dependency checks stay enabled. For the Bazel build, verified upstream
archives can be preloaded in `extern/bazel-distdir` under their original
filenames. Bazel still checks its locked digests.

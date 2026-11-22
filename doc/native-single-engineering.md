# Source-bound native single-ring engineering path

Inputs are the actual MLIR bytes and a secret input port spec with a finite
domain. The native importer and Pass 1 bind periodic public matvecs and the
original activation SSA. `NativeArithmetic` propagates affine ranges, selects
certified SiLU/erf-GeLU DAGs and expands the chosen route to mask, rotation,
rescale, multiply and add edges. `specializeNativeArithmetic` tests schedules
against the same Pass 3 budgets. Its result is *not* security approval.

`NativePhysical` replays the selected integer basis and the actual per-call
encoding branch with `ExactScaleTracker`, accounts for both sides of a join
and rechecks headroom and the sink engineering allowance. Unsupported scale
preparations or op types fail closed. This path is one periodic ciphertext,
one P, exact key switches (`k=0`), and one-Q rescale. It does not select a
dynamic ring boundary or produce a probabilistic noise certificate.

`NativeSingleGo` checks the trusted pinned-oracle report (ordered Q/P, Q,
QP, envelope, ring and assumptions) and reconstructs the physical plan before
emission. It emits a self-contained evaluator plus the runtime source, with
separate seals for plan semantics and exact JSON bytes. The runtime accepts
the checked parameters and evaluation keys, not a different profile with the
same bit widths. `EncodeInput` enforces the declared domain on plaintext
input; ciphertext-only callers are responsible for the input contract.

On the project server only, use a private checkout, pinned Go/MLIR/HiGHS,
Sage with the pinned lattice estimator, strace and Python with mpmath:

```sh
export SNABF_EXTERN=/home/fhecompiler/extern
export PATH="$SNABF_EXTERN/go/bin:$PATH"  # also put pinned strace on PATH
export GOPATH="$SNABF_EXTERN/gopath" GOPROXY=off GOSUMDB=off GOWORK=off GOTOOLCHAIN=local
R=$(mktemp -d); export GOCACHE="$R/go-cache"
cmake -S src/compiler -B "$R/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$SNABF_EXTERN/highs-install" -DHEIR="$SNABF_EXTERN/heir" \
  -DSNABF_MLIR_ROOT="$SNABF_EXTERN/mlir24/usr/lib/llvm-24"
cmake --build "$R/build" --parallel 8
ctest --test-dir "$R/build" --output-on-failure
EXE="$R/build/ir/NativeArithmeticTest"
python3 -B test/compiler/phase2_native_security.py "$EXE" "$R/security" --estimator "$SNABF_EXTERN/lattice-estimator"
python3 -B test/compiler/phase2_native_cipher.py "$EXE" "$R/security" "$R/cipher"
```

The engineering driver calls the real C++ checker and emitter; `--check` here
is not the MLIR `--snabf-check-plan` pass. For the public CLI, MLIR passes
and independent product gates, see `src/tools/native_n1.md`.
The private harness checks known source hashes, per-PID compiler traces,
coordinated mutations, module dependencies, immutable package files and every
complex decoded slot against independent source truth. Do not publish its
measurements, cache or working directories. Do not report Linux as Windows.

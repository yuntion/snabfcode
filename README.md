# SN-ABF compiler

Security-Noise Coupled Algebraic Budget Flow on Google HEIR and Lattigo.

### Static compiler entry (phase 2)

On the configured Linux toolchain, build `snabf-opt` as for the native suite and point `SNABF_OPT` at that binary. The product entry accepts **one static packed f32 vector in/out**, constant matvecs, source-marked SiLU and residual addition. It does not consume a workload name, frozen corpus, hand-authored parameter file or an old plan. `--log-n`, `--q-count` and `--min-slots` specify the provisional HEIR typing capacity. Automatic deep bootstrapping is not in this subset. The source interval is a caller contract.

```bash
python3 src/tools/snabf_compile.py --mlir /path/to/model.mlir --input-bound 1 \
  --precision-bits 12 --security-bits 128 --mode keep-source --emit-go --out-dir /tmp/new-model
python3 test/compiler/phase2_static_smoke.py  # generates 3 matvecs, 2 SiLUs and a residual; encrypts and checks every slot
python3 test/compiler/test_phase2_entry.py -v
```

Use a **new** output directory. The compiler publishes `work/checked.mlir`, `compile.json` and, with `--emit-go`, `model.go` and `program.json` together after `go build` succeeds. Cache hits are owner-private local estimator measurements, not signed approvals. On a miss, `sage` and the pinned `LATTICE_ESTIMATOR` checkout (by default under `extern`) must be available. `adaptive` requires net benefit; `force-down` requires a verified, not necessarily profitable switch. Set `SNABF_PASS4_ESTIMATES` to the compiled `work/domains.json` for replay, or unset it for keep-source. `single-live-v1` (default) preserves the old crossing-residual rejection. Opt in to `--cut-profile dual-live-v1` for at most two live values on one periodic boundary with a terminal linear or residual-add suffix. This is not arbitrary multi-ciphertext repacking. No probabilistic certificate is issued. See [phase 2 decisions](doc/phase2-decisions.md).

For the three-arm encrypted regression, pass **only the source MLIR** (no checkpoint, cached plan or frozen truth). The runner derives a common corpus from the source, compiles each arm, checks all slots and writes `acceptance.json`, negative gate results, and `SHA256SUMS` in a new output directory:

```bash
python3 test/compiler/phase2_dual_live.py --source /path/to/cross_residual/source.mlir --out /tmp/dev2-new-run
```

The first full experiment is specified in [E01](doc/experiment-01.md).
The fixed-profile control and [native Pass 1](doc/pass1-revision/results.md) run end to end.
The primary compiler is now the native [SN-ABF dialect and C++ pass pipeline](doc/unified-ir-development.md).
Start with [SNABFDialect.td](src/compiler/ir/SNABFDialect.td), [the operations](src/compiler/ir/SNABFOps.td),
[Passes.td](src/compiler/ir/Passes.td), or [an actual selected module](doc/unified-ir-results/reuse64-joint.mlir).
The Pass 1 suite compares stock HEIR, independent routing and graph-wide key
reuse on three encrypted SiLU blocks. It also runs a separate end-to-end periodic-layout
ring-switched bootstrap, residual precision feedback and concrete Pass 3 specialization.
Run `bash src/tools/run_pass1.sh` for the C++/MLIR and ciphertext regressions.
Pass 1 has no Python optimizer or text-IR fallback.
This is not yet the four-pass E01 result.

The native [Pass 2 implementation and development report](doc/pass2-development.md)
extends that same planner with certified Remez stages, full activation DAGs,
and downstream feedback. [Encrypted results and decision records](doc/pass2-results/results.md)
cover the original models, a residual network and explicit composite-template coverage.
On the configured server run `bash /home/fhecompiler/src/tools/accept_pass2.sh`
for a fresh build, C++ tests, real encrypted acceptance and a new evidence archive.
The mathematical core requires the MPFI, MPFR and GMP development libraries.

The native [Pass 3 implementation](doc/pass3-development.md) selects concrete
RNS parameters and centered-prefix gadget sites, re-specializes HEIR, and
checks the actual Q/P before lowering. Its [encrypted comparison](doc/pass3-results/results.md)
includes independently optimized exact, joint, and same-ciphertext matched
exact runs on `reuse64` and `residual`. Entry point: `src/tools/run_pass3.sh`.

```text
doc/          Experiment protocol and measured integration results
extern/       Pinned HEIR, HiGHS, Go, Bazel and Lattigo source (extern/lattigo)
src/
  frontend/   SN-ABF extensions to the upstream HEIR frontend
  compiler/   ir/ and pass1/ through pass4/ implement native planning; Target/Lattigo/ is shared
  backend/    Lattigo kernels with explicit layout and level contracts
  tools/      Linux setup, compilation, and test entry points
test/         MLIR and profile fixtures, C++ regressions, encrypted Go tests
build/        Generated MLIR/Go, run manifests, and logs (not source)
```

On the Linux server, the dependencies are already under `extern`. On a clean
Linux x86-64 checkout, follow [the dependency instructions](extern/README.md).

The [Lattigo target](src/compiler/Target/Lattigo/README.md) documents which code
is emitted by upstream HEIR and which calls are bound to SN-ABF runtime kernels.

```bash
cd /home/fhecompiler
bash src/tools/bootstrap_mlir.sh  # first installation only, Ubuntu 22.04 x86-64
bash src/tools/run_pass1.sh       # current native suite; fresh build/pass1-revision/run.*
bash src/tools/run_reference.sh
```

The frontend calls `HeirOptBackend` and `HeirTranslateBackend` from
`extern/heir/frontend/heir/heir_cli`. It imports the project's MLIR, freezes the
reference `Q/P` through HEIR's parameter-file option, and writes `input.mlir`,
`lowered.mlir`, `model.go`, and `manifest.json` to `build/baseline/`.
The Go test executes that generated program on encrypted inputs and compares
the decoded outputs against an independent float64 SiLU reference. Results
are emitted as `SNABF_METRICS` JSON in `build/logs/reference-tests.log`.

## Frontend extension point

Keep local importers and model-specific annotations in `src/frontend`, without
editing `extern/heir/frontend`. `snabf_frontend.compile_reference` accepts an
`importer: Callable[[Path], str]`: return `linalg`/`secret` MLIR with the secret
arguments and activation domains annotated, and HEIR handles actual parsing,
verification, and lowering. The checked-in input is hand-authored in the
Torch-Linalg input form. A Torch/ONNX importer can use the same interface.

`HEIR_SOURCE_DIR` selects the upstream Python frontend and `HEIR_BIN_DIR`
selects the compiler binaries, so the project frontend can use a source-built
SN-ABF-enabled HEIR once the dialect is registered. The current reference arm
accepts plain HEIR IR and does not claim SN-ABF rewrites.

```bash
bash src/tools/test_heir_source.sh
```

This second command tries two targeted upstream Bazel tests. See
[server validation](doc/validation.md) for the results and fetch status.

## Pass 4: mixed-ring CKKS inference

`bash src/tools/run_pass4.sh` rebuilds the native passes, independently screens
concrete Q and QP instances with the pinned Sage estimator, and runs the
baseline, adaptive, and expensive-conversion arms on `reuse64` and `residual`.
It uses the archived Pass 3 inputs and float64 oracle, **not** their outputs
or timings. Each run creates its own `build/pass4/run.*` directory. The
multi-epoch C++ emitter lives in `src/compiler/Target/Lattigo`; Lattigo key
generation and encrypted execution live in `src/backend/ringswitch/epochs`.
The output ABI returns the checked sink in its actual key domain. A residual
that crosses the proposed boundary is evaluated in the retained source ring.
Full-packing split/repack and generic up-conversion are not
advertised as executable for this ABI.

`bash src/tools/run_pass4_ab.sh` compares the native C++ Go emitter to the
historical `go-bind` on the same four sets of ciphertexts and evaluation keys.
It separately tests a target-domain result, a residual crossing the last
linear layer, and a switching-key security rejection. The last two must run in
the source domain without a conversion key. The pinned HEIR translator emits
the shared ordinary CKKS functions; the online model calls them from straight-line
Go alongside the selected SN-ABF kernels. The reproduction does not edit that Go.

## Phase 2: multi-P search, exact GeLU and noise audit

The static source entry recognizes the supported linear, residual, SiLU and
exact erf-GeLU patterns before upstream lowering. It does not accept arbitrary
ONNX operators or replace erf-GeLU with tanh-GeLU. New run directories must
not already exist. On Linux with the pinned dependencies:

```bash
cd /home/fhecompiler
export SNABF_OPT="$PWD/build/native/ir/snabf-opt"
export LD_LIBRARY_PATH="$PWD/extern/mlir24/usr/lib/llvm-24/lib:$PWD/extern/mlir24/usr/lib/x86_64-linux-gnu:$PWD/extern/highs-install/lib:${LD_LIBRARY_PATH:-}"
run=$(mktemp -d)
python3 test/compiler/phase2_gelu_e2e.py --out "$run/gelu" --security-cache "$HOME/.cache/snabf/measure-v1"
python3 src/tools/verify_e2e_pass1_4.py "$run/gelu" --c1-gelu
python3 src/tools/replay_noise_audit.py "$run/gelu/mixed/model" --snabf-opt "$SNABF_OPT"
python3 test/compiler/phase2_multip_ablation.py --mlir /path/to/reuse64.mlir --out "$run/multip"
python3 test/compiler/phase2_multip_ablation.py --replay "$run/multip"
```

`--gadget-alpha 1..4` selects a fixed P count, `0` searches `{1,2}`; omitting
it (with no heterogeneous flag) preserves legacy single-P plans. Partly zero
CRT digits still require an external product. `--heterogeneous-scales` in the
source compiler enables the wider search **within the pinned HEIR integer
rescale bin**. The standalone native solver can search without this constraint;
such a result is not automatically HEIR-emittable. Inspect the actual non-base
Q spread, measured transforms and serialized bytes before interpreting a run.
The continuous bounds `--prime-bw-min`/`--prime-bw-max` default to 24/59;
q0 retains its own floor. `--aux-prime-margin-bits` defaults to two; larger
margins are checked against the products of the actual integer primes.
Only `joint` and `matched_exact` share a ciphertext, concrete basis and keys.

By default, `audits/noise_cert_audit.json` reports known and missing primitive
contracts; it is not a proof of the observed accuracy or a failure probability.
`--certify-failure-bits 40` requires a complete analytic certificate: without
one the compiler returns nonzero, retains a denial audit next to the requested
output, and publishes no model. Classical estimator screening remains separate.

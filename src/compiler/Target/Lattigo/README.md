# Lattigo Target

The `ring-go` target emits one Go file in package `planned`. `planned.New()`
binds the concrete key families and encoded constants and returns an
`*epochs.Runtime` with `Encrypt`, `Eval`, `Decode` and `SourceEvaluation`.
`Eval` calls the model-specific straight-line Go function. The JSON sidecar is
an audit, never an execution input. The model has no dependency on `go-bind`.

## Build and emit

From the project root, using the pinned HEIR/MLIR, HiGHS and Go environment:

```sh
cmake --build "$build" --target snabf-opt
opt="$build/ir/snabf-opt"
# Pass 3 only: input.mlir must have the concrete profile and its security screen.
"$opt" --allow-unregistered-dialect --snabf-check-plan \
  --snabf-lower-ckks=direct-go input.mlir -o lowered.mlir
# For Pass 4 use the existing --snabf-lower-ckks instead, on the output of
# --snabf-pass4. Use the same SNABF_PASS4_ESTIMATES for each invocation.
"$opt" --allow-unregistered-dialect \
  '--snabf-bind-lattigo=ring-go=model.go audit=program.json' lowered.mlir -o /dev/null
go build -mod=readonly model.go
```

The Go workspace is the project module `snabf.local/compiler` (or an external
module requiring it with a local `replace`). `model.go` imports
`src/backend/ringswitch/epochs`, the HEIR `heirckks` subpackage, and `gadget`
where needed. Put separate models in separate directories/packages; do not
merge or patch Go sources. The public `SourceEvaluation()` returns the *live*
`(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder)`, never secret keys or the
decryptor. Its scratch buffers are not for concurrent use with `Eval`.
For legacy equality tests encrypt once, use `ct.CopyNew()` for each path and
the same runtime for decoding. This works for single-domain output; `Decode`
checks the **target** domain. A switched suffix is not a source-domain result.

## Boundaries

* `EmitVerifier.cpp`: `snabf-check-plan` replays and issues `snabf.emit_sha256`.
  The digest binds generic, location-free SSA including profiles, signed
  payloads, external-screen records, decisions and cut inventories. Lowering
  and emission independently replay the source, check the digest, and compare
  the reconstructed lowering. A hash is a consistency check, not an attestation
  by a trusted signer. Historical IR without a receipt must be rechecked.
* `RingEmitter.cpp`: pairs physical Pass 3 edges with the selected tower's
  signed HEIR diagonal tables and the actual Pass 2 coefficients. Buffer
  domains come from the verified graph, not textual order around a conversion.
* `EmitGraph.*`: checks topological order, arity, operand domains, levels,
  geometric rotation-to-Galois inventory and byte count. A down edge must bind
  to the sole converted live operand, its layout and the screened extended QP.
* `DirectGo.cpp`: emits the typed `epochs.Program` literal and one static Go
  call per selected edge. Hoisted babies share one call per (route site, SSA
  input), preserving every rotation's `K`. The nonlinear DAG, scalar encodes,
  residual drops, relins and rescales are materialized, not refitted here.
  Go verifies actual ring, level, physical scale and slot shape at every edge.
* `StandardCKKS.mlir` supplies HEIR's ordinary Add/AddPlain/Mul/MulPlain/Rescale
  functions. `scheme.ckks` is on the physical module. `standard.go` is the
  unmodified output of pinned `heir-translate --emit-lattigo --package-name=heirckks`;
  do not gofmt it. The static level-drop count uses `DropLevelNew` directly.
  Upstream errors from helpers are returned by `Eval` with their concrete site.
* The target stages both files before publishing, replaces each by rename,
  and publishes Go last. Use one writer per output path. A concurrent reader
  should match `planned.PlanSHA256` to `emit_sha256` in the sidecar: a pair of
  ordinary files is not a filesystem-wide transaction. Failed checks preserve
  existing output files.

## Verification

```sh
HEIR_TRANSLATE="$heir/heir-translate" bash src/compiler/Target/Lattigo/regenerate-heir-helpers.sh --check
go test -mod=readonly -count=1 ./src/backend/ringswitch/epochs/...
python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" current.mlir
SNABF_PASS4_ESTIMATES="$security" python3 src/compiler/Target/Lattigo/test-direct-gate.py "$opt" current.mlir --pass4
```

Use `--write` instead of `--check` only to regenerate the public helper. With
the locked `heir-py 2026.11.1` translator the raw helper SHA-256 is
`3ba17babf49e6153724b374ebc22232f047121a9094c841fa8512c8c8d3f3102`.

The direct ABI supports one periodic vector, one sink and, when selected,
one checked trace-invariant down conversion. Missing concrete Pass 3 records,
arbitrary packing, multi-ciphertext repacking, and bootstraps inside this
epoch graph are not silently translated. The separate historical Pass 1
periodic bootstrap and legacy HEIR comparison scripts still use `go-bind/`;
the stock HEIR control remains unmodified. A successful emission is not
an end-to-end error probability proof or a performance result.

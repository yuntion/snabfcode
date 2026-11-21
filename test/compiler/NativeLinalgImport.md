# C-1 source inspection

Build target `snabf-native-import` in the normal pinned-SDK CMake project:

```sh
"$build/ir/snabf-native-import" SOURCE.mlir --input-spec SPEC.json -o RECEIPT.json
```

Without `-o`, the receipt goes to stdout. With `-o`, the destination must
not already exist. Rejections exit nonzero, diagnose on stderr and create no
receipt. The C++ API is `snabf::native::inspectNativeLinalg` in
`NativeLinalgImport.h`; it accepts two byte buffers and a diagnostic source
name. Neither the library nor the CLI calls HEIR, compiles a model or selects
keys. The library target links the pinned `libMLIR` and `libLLVM` directly.

```json
{"schema":"snabf.native-input-spec.v1","inputs":[{"func_symbol":"linear_silu_linear","arg_index":0,"source_id":"x","classification":"secret","domain":[-1,1]}]}
```

Use the bare MLIR function symbol (not `@...` or an SSA argument name).
C-1 accepts a single secret vector input and a single vector result, both
with the same static periodic logical width. The classification must be
explicit even when the argument bears the optional `secret.secret` unit
attribute. `domain` is an optional declared closed interval of finite
binary64 endpoints. Omission is reported as null, never as `[-1,1]`. A
source activation's declared range does not establish an input domain or a
range proof. Neither IDs nor different sites imply statistical independence.

The receipt schema is `snabf.native-import.v1`, `frontend` is
`native-linalg`, `stage` is `structural-inspection`. `source_sha256` and
`input_spec_sha256` hash the raw bytes passed to the API, including comments,
whitespace and line endings. `func_symbol`, `inputs`, `outputs`, `nodes`
and `activation_locks` come from verified MLIR. No model name selects a
topology. Inspect this receipt as a source witness, **not an executable
plan**. Verifiers must rebind to trusted source and spec bytes, not just
cross-check derived fields in an untrusted receipt.

| Field | Interpretation |
| --- | --- |
| `inputs[]` | `(func_symbol,arg_index)`, value `id`, source ID, classification, whether the marker was present, declared `domain` or null, logical type and periodic width. No physical slot count or ring is inferred. |
| `outputs[]` | `(func_symbol,result_index)` and `value` referring to the returned top-level definition, with logical type, period and return location. |
| `nodes[]` | Lexical/topological order; definition `id`, source `op`, checked `kind`, `logical_type`, physical `location` (one-based line/column) and **ordered** `inputs`. `arg0` is the input port, not a node. |
| `constant` | `shape`, `element_type` and `bits`: an expanded row-major array of fixed-width lower-case `0x...` IEEE words, including the sign of zero. Scalars have shape `[]` and one word. |
| `matvec` | Inputs are `[matrix,data,initial_accumulator]`; `outs` repeats the last ID. The row-wise scalar body, indexing and positive-zero accumulator were checked. `tensor.empty` alone is not a zero accumulator. |
| `activation_locks[]` | `kind` (`silu` or `gelu_exact_erf`), outer `site`, `x`, `result`; scoped `region_x`, `region_result`, and ordered scalar `body` with actual SSA inputs. `literals` bind roles to constant IDs, types and exact bits. `declared_domain` is not a proof. `origin` binds the source digest, function, and empty call/iteration paths: no inlining, unrolling or folding occurred. |

The scope is intentionally narrow: one block, a matching power-of-two vector
width up to 1024, f32/f64, finite public dense weights, square matvec, and
defined elementwise add/sub/negate or one-public-operand multiply. With the
original generic spellings, `outs(tensor.empty)` is valid only because the
output block argument is unused. Other generics, dynamic control, calls,
nonzero accumulators, fast math, ReLU and arbitrary dialect attributes are
not imported. Limits here are frontend admission rules, not parameter or
precision claims.

## Reproduction

```sh
cmake --build "$build" --target snabf-native-import NativeLinalgImportTest
ctest --test-dir "$build" -R NativeLinalgImport --output-on-failure
python3 -B test/compiler/NativeLinalgImportSources.py "$build/ir/snabf-native-import" \
  --c1-source "$original_c1_single" --strace "$strace" --keep "$new_receipts_dir"
python3 -B test/compiler/NativeLinalgImportSources.py --replay "$new_receipts_dir"
```

The two example specs and the original C1 source-only member are in
`test/compiler/native_import/`. C1 was taken from `c1-e2e/single/source.mlir`
in `gelu-full.tgz` (archive SHA-256 `7c5692b61c140f9677adc93684ee44a35b0b7ec0f06cefbda774df5c880a9306`;
member `69f3b56c72c261a37a1c229ba8925fa6d07b6da2b3b39240cf4dcd3f7b294ec4`).
The independent test oracle checks that member and all three R3 source
digests before parsing text. On a CRLF checkout it obtains an R3 original
with `git cat-file blob` at the pinned baseline; it never normalizes a
modified file and calls it the source. The CRLF spelling is separately
inspected as a *different*, hash-bound source. `--replay` needs no MLIR SDK.

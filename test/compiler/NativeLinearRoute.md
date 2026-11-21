# Native square matvec to Pass 1

`snabf-native-linear` reads the original MLIR and `snabf.native-input-spec.v1`
bytes. It emits `snabf.native-linear.v1` to stdout only after verifying the
whole graph. This is a candidate catalog, not a selected or encrypted program.

```sh
cmake -S src/compiler -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DSNABF_MLIR_ROOT=/home/fhecompiler/extern/mlir24/usr/lib/llvm-24 \
  -DCMAKE_PREFIX_PATH=/home/fhecompiler/extern/highs-install -DHEIR=/home/fhecompiler/extern/heir
cmake --build "$build" --target snabf-native-linear NativeLinearRouteTest -j3
"$build/ir/snabf-native-linear" SOURCE.mlir --input-spec SPEC.json --log-n 14 --encoded-slots 4096
ctest --test-dir "$build" -V -R NativeLinearRoute
```

The surrounding CMake project configures other targets; neither this CLI nor
its library links a HEIR target. There is no legacy import, generated CKKS
type, synthesized HEIR interface, external program invocation or JSON-to-IR
path. Rejections write diagnostics to stderr, with no receipt on stdout.

## Source and physical map

`inspection` is the C-1 receipt from the **same parse**, including raw source
and spec SHA-256, input/output ports, SSA edges, activation locks, and source
locations. The callback sees only checked `linalg.matvec` operations with
constant rank-two square weights, row-by-column reduction, and positive-zero
fill or dense `outs`. A later failure (even after the last matvec) discards
all derived sites. An inspection with no matvec cannot produce a route catalog.

`layout` describes one explicit domain for all sites. With `N=2^logN`,
`G=N/2`, `E=encoded_slots` and `w=logical_width`, require `w | E | G` and
powers of two. The map for **both** input and output is logical row `i` at
`first_slot_for_row[i] + k * replica_stride` for `0 <= k < replica_count`.
Here these are `i`, `w`, and `E/w`: one tiled vector, no unused encoded slots
and no independent samples in the replicas. `G` is the physical automorphism
group order, not a claim that `G` slots were encoded. For example with `N=16384`,
`E=4096`, `w=4`, left offset `-1` is physical `8191`, encoded `4095` and
logical residue `3`. Explicit domain arguments describe the required packing;
an MLIR tensor shape does not establish that an arbitrary ciphertext has it.

`matvecs[]` is in verified source order. `weight_id`, `input_id`, `outs_id`,
and `result_id` are definitions in `inspection`, not printed SSA names. Follow
the graph's ordered edges to the input and returned ports, including residual
branches. `outs_kind` records the proven positive-zero origin; there is no
uninitialized accumulator or implicit bias. `weight_sha256` hashes the expanded
row-major IEEE words of the actual source literal, big-endian at `element_bits`
(32 or 64), with no conversion to host floating point. The weight's shape,
element type and definition are bound by that source graph as well.

For `Rot_d(x)[j] = x[(j+d) mod E]`, the unadjusted diagonal is
`D_d[j] = W[j mod w, (j+d) mod w]`. Choose the centered signed residue, with
`-w/2` at an even tie (`w=1` has only offset zero). Each `diagonals[]` entry
contains `signed_offset`, `physical_offset = d mod G`, `encoded_offset = d mod E`
and fixed-width hex `row_bits` for one output-row period. Omit only all-**+0**
diagonals; even a negative-zero word is preserved. Empty support is the zero
linear map, not an identity transform. These are mathematical linear maps;
this stage does not certify an arbitrary f32/f64 reduction's roundoff.

## Pass 1 consumer

The `snabf_native_linear` C++ target exposes `deriveNativeLinear` and
`NativeLinearProgram` in `src/compiler/ir/NativeLinearRoute.h`. The per-site
`candidates` are the actual `std::vector<snabf::SubgroupRoute>` returned by
`executableRoutes(logN, signedOffsets)`; a native planner can supply these as
`LinearRegion::candidates` without reading JSON. The test harness also sends
the four graphs' typed candidates to `selectSubgroupRoutes` with test-only
costs. No test cost, key scope or selected index is in the production output.

`route_candidates[].terms` are `[physical_offset, baby, giant]` in `Z_G`.
`equivalent_splits` records `[rung,stride,order,index]` for folded alternatives;
`baby_rotations`, `giant_rotations` and `galois_keys` are requirements, not
materialized keys. Join terms to diagonals by **physical offset within the
same site**, not by ordinal or by encoded offset. For a chosen candidate call
`program.materializeMask(site, candidate, term, span)` with exactly `E` words.
It emits `Rot_-giant(D_d)` so the actual operation is
`Rot_giant(mask * Rot_baby(x))`. A returned word is an f32/f64 bit pattern,
not a converted binary64 value or an encoding receipt.

## Validation and remaining integration

`NativeLinearRouteSources.py` fixes the four source hashes, reading Git blobs
when a Windows R3 checkout differs from the archive. No newline normalization.
The independent Python oracle reads matrix columns and source SSA bindings;
it checks layout, centered and physical offsets, split coverage, Galois keys,
actual C++ mask bytes and every encoded slot for every logical basis vector
and candidate. Separate negative tests bypass bit comparisons and require the
basis action itself to reject a wrong giant orientation and a wrong last
replica. Same-column action alone cannot detect an encoded-equivalent but
physically different key; there is a coordinated wrong-lift/route test for it.

```sh
python3 -B test/compiler/NativeLinearRouteSources.py "$build/ir/snabf-native-linear" \
  "$build/ir/NativeLinearRouteTest" --save /path/to/new-witness-dir
python3 -B test/compiler/NativeLinearRouteSources.py --offline /path/to/new-witness-dir
```

For N1 integration, bind the same source/spec and a trusted **per-epoch** ring,
encoded slot mapping and key domain. Re-derive physical lifts and candidates on
domain changes. Populate real key scope (including ordered Q/P), levels, key
sizes, calibrated prices, and applicable path/depth cuts before selection. The
catalog folds equivalent programs for occupied pricing; for dense-tower pricing
re-enumerate with `foldEquivalent=false` from the verified signed offsets.
Downstream must implement the zero map at a defined level/scale, SSA allocation,
actual encoding and scale receipts, activation lowering, and decode-domain
checks. No mixed-ring policy, ciphertext accuracy, ReLU release, or DirectGo
execution is implied by a successful matvec basis test.

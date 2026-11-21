# Native source inspection and package checks

`legacy-heir` remains the default. A native inspection is a source witness,
not a parameter plan, security decision, proof of approximation, or encrypted
run. The normal native execution request remains fail-closed until the N1
lowerer, four-graph numerical gate, and independent scale checks are ready.

With the pinned `snabf-native-import` built next to `snabf-opt`, inspect one
static source into a new directory:

```sh
python3 src/tools/snabf_compile.py --frontend native-linalg --native-import-only \
  --mlir source.mlir --input-spec input.json --input-bound 1 \
  --snabf-opt /path/to/build/ir/snabf-opt --out-dir /tmp/my-inspection
```

Use a `snabf.native-input-spec.v1` object with an `inputs` array, an explicit
`func_symbol`, `arg_index: 0`, `source_id`, `classification: "secret"`, and
the declared `domain: [-1, 1]`. The inspection publishes only `import.json`.
It hashes the original source and input spec as bytes. In particular a CRLF
checkout is not interchangeable with an LF archive.

`test/compiler/phase2_n1_equiv.py inspect|compile` accepts `--r3-frozen`,
`--single-gelu`, `--snabf-opt` and a fresh `--out`. The source inputs are
pinned to R3 and C1, not inferred from a fixture or model name. On Linux the
probe requires `strace -ff`; an unresolved exec or missing trace fails.
`slots` replays 64 raw trial rows against source-derived truth, including
every periodic output slot and the strict `< 2^-12` threshold. These probes
do not assert that two outputs share a key, ciphertext, or decoding domain.

`observe GRAPH SOURCE MODULE --out DEST --seal SHA --ring N --level L
--slots COUNT --scale EXACT` runs `New/Encrypt/Eval/Decode` on a disposable
copy of the built package. It requires the independently checked sink ABI.
The runner records the actual ring, level, binary64 and full rational scale,
serialized switch-key bytes, evaluation time, and every decoded slot.
Successful runs publish `trials.jsonl` and `run_observations.json` together in
`DEST` outside the module. Missing slots, an error equal to the threshold,
or a mismatched exact scale reject the run even if binary64 rounds identically.

`native_package.publish_native_package` is an internal post-emission gate;
its caller must supply the checked IR seal. It uses the pinned Go and Lattigo
versions and an isolated module with `GOWORK=off`, tests every package, and
publishes only after checking that build/test left the files unchanged. The
seal is **not** the file hash of `program.json`. `package-build.json` records
both digests. Keep run observations outside the published module. This gate
tests package construction only; it cannot authorize an arbitrary MLIR graph
or issue a probability certificate.

To attribute a 17-Q search refusal independently, run
`test/compiler/phase2_a11_17q_attribution.py --prior ORIGINAL_A2A --opt OPT
--out NEW_DIR`; use `--replay NEW_DIR` instead of `--opt/--out` offline. The
pre-solver guard uses the decoded magnitude envelope from the selected DAG.
A refusal of this envelope is not a proof that the graph cannot execute with
any concrete 17-prime parameterization.

The separate `test/compiler/phase2_fullslot_linear_runtime.sh PASS1 NEW_DIR
[OPT]` probe exercises a real one-to-two full-slot split, four nonempty target
linear blocks, the joint rotation-key inventory, and reconstruction by logical
index. Provide the pinned Go toolchain and offline module cache. This is a
runtime component test; the production Pass 4 profile, three whole-graph arms,
concrete security admission and feedback loop are not enabled by it.

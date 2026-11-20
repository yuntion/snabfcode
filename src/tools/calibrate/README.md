# Whole gadget site calibration (A2.1)

Build in the measured checkout with the locked Go and Lattigo. Use a new run
directory and reserve a Linux CPU. The tool checks affinity; it cannot detect
or guarantee host exclusivity. Do not use a WSL smoke profile for acceptance.

```sh
go build -mod=readonly -o "$RUN/snabf-calibrate" ./src/tools/calibrate
HW=$(taskset -c "$CPU" "$RUN/snabf-calibrate" --print-hardware-id)
taskset -c "$CPU" "$RUN/snabf-calibrate" --manifest "$RUN/cases.json" \
  --warmup 5 --samples 30 --holdout 5 --out "$RUN/hw_profile.json"
```

Without `--manifest`, the first runnable slice measures N=8192, alpha=2,
ell=2: relin, giant(1), and one shared baby group (1,2). This is not the
full grid. A manifest has this shape (replace the example basis with the
**complete ordered integer** Q/P for each actual candidate; do not round-trip
through binary64). One case per basis/ell/op/ordered shifts. Omit neither
the all-native pattern nor any combination that should be eligible.

```json
{"cases":[{"n":8192,"q":[33538049,68719230977,68720050177],"p":[549755731969,274878136321],"ell":2,"op":"baby","shifts":[1,2],"patterns":[[0,0],[1,0],[1,1],[2,2]]}]}
```

`relin` uses `shifts:[]`; `giant` uses one nonidentity shift. Both use
singleton k vectors. A baby case is one emitted `Babies` call, including
native/cross-k hoist sharing, **not** a per-consumer price. Shifts come from
the emitter's `(route_site,input)` groups in emitted order. `0<=k<=min(ell,4)`.
Keys are expanded pure RNS, at the full basis; input is standard-ring CKKS,
batched, non-Montgomery NTT. Plaintexts, ciphertexts, keys, evaluator and
site descriptors are prepared before timing. Allocations and copies inside
the actual ABI (in particular Giant's input copy) are charged. Each pattern
is timed in rotating order; the holdout uses a separately encrypted input.
Timing, dense big-integer projection checks, and observer counts are distinct
invocations. No reference replay or keygen is in the measured wall time.

Schema `snabf-gadget-sites-v1`: samples and quartiles are **decimal strings in
microseconds** (5 fractional digits; raw clock resolution is nanoseconds).
Training median/IQR use type-7 interpolation. The independent holdout never
sets prices. `observed_non_native` has no k=0 entry: the native runtime emits
no observer event. The three observed counters are separate, not conversions.
`hardware_id` hashes the hardware map including affinity, microcode, governor
and hashed machine-ID (or an explicitly boot-scoped ID on hosts without one).
`bench_rev`, the actual binary hash and Go build settings identify the tool.
`profile_sha256` hashes the canonical payload excluding only itself: objects
are `{` plus key/value pairs sorted by UTF-8 key bytes plus `}`; arrays are
`[` plus ordered values plus `]`; strings are `s<UTF-8-byte-count>:<bytes>`;
integers are `i<decimal>;`. No nulls, floats or booleans. Integrity is not
attestation of who acquired the measurements or of cryptographic security.

C++: `o.costProfile = CostProfile::loadFile(path, HW)` with explicit
`o.alpha` (0 for auto). `SitePricing.cpp` belongs to `snabf_pass3`;
`CostProfile.cpp` and `Pass3CalibrationTest.cpp` need LLVM Support. Null
`costProfile` preserves legacy proxy behavior. Calibrated source search is
restricted to measured physical bases, with original constraints and concrete
replay. Missing k vectors are disabled, missing native and DownConv pricing
are rejected. Receipts bind the profile hash, hardware, ordered edge members
and group medians; `gadget_objective_us` is the sum of gadget charges, **not**
predicted end-to-end time. For the default Go profile, an optional live test
is `Pass3CalibrationTest "$RUN/hw_profile.json" "$HW"`.

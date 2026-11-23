# Native single-ring product profile

Configure `src/compiler` with the pinned MLIR SDK and `-DSNABF_NATIVE_ONLY=ON`;
build `snabf-native-opt`. This target does not configure or link the HEIR dialect.
Run `snabf_compile.py --frontend native-linalg --mlir ORIGINAL --input-spec PORTS
--input-bound B --snabf-opt /absolute/path/to/snabf-native-opt --security-cache PRIVATE
--emit-go --out-dir NEW`. The single, periodic port must declare `[-B,B]`.
The supported product profile uses N=16384, 4096 encoded slots, precision 12,
a single P and exact (k=0) gadget arithmetic. Adaptive selects keep-source;
forced ring down and probabilistic certification are not supported.

`planned.mlir` carries the original source and port bytes and the physical
plan. A verifier must supply its own trusted `native-security` receipt (not a
path from the module). The CLI obtains it from the owner-private pinned
estimator cache or the estimator itself, for the concrete ordered Q, QP and
envelope requests. Invoke `--snabf-check-plan=native-security=security.json`
on the planned or checked module. Separately invoke
`--snabf-lower-ckks=native-security=security.json native-output=NEW` on the
checked module. Each rederives the plan from the embedded original bytes and
checks the security context. A digest or a checked stage tag is insufficient.
The CLI publishes these MLIR stages and a self-contained `module/` with the
generated `planned/` package. Without `--emit-go` it stops after checking.

The server gate `test/compiler/product_native_n1.py` takes the native opt,
private measurement cache and a new directory. It runs the public entry, both
negative gates, full-module Go build/test, and fresh encrypted trials. The
portable batch contains only the four original sources, per-trial real and
imaginary slot observations, and their manifest. Independently supply the
manifest SHA-256 to `verify_n1_equivalence.py BATCH SHA256` on the approved
offline machine. Do not publish private observations, traces, or caches.

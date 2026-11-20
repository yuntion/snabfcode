# Server validation

Run on November 18, 2026, on the AMD EPYC 9754 Linux server. The paper's
Appendix SEAL case study is a separate result and is not used as a substitute
for this implementation.

| Check | Result |
|---|---|
| `src/frontend` extension tests | Custom importer output lowered to the renamed Go entry; invalid importer MLIR rejected by HEIR |
| Frontend -> official HEIR -> generated Go -> Lattigo -> decoded `Linear -> SiLU -> Linear` | 64 ciphertexts, 256 checked outputs, all within `2^-12`; final run had `L_inf = 9.15e-5`, `MSE = 2.40e-10` |
| Actual reference profile | `N=16384`; zero-based level `7 -> 0`; measured `log2(scale)` `26 -> 51.96`; exact `Q/P` in the checked-in profile |
| Lattigo `DownConv` | `N=16384 -> 8192`, eight complex slots, unchanged active level and scale; final run had map `L_inf = 4.24e-10` |
| Independent packed values | Native projection averaged `0.75` and `-0.25` to `0.25`; the layout-preserving kernel rejected that full-packed input |
| C++ routing and HiGHS | Subgroup/coset and minimum-cost key-union checks passed; sharing the same physical key changed the optimal first-layer route |
| HEIR source Bazel tests | Did not reach compilation. `rules_python` was resolved with a matching upstream archive; other GitHub release dependencies still time out (`bazel_skylib`, `rules_cc`, `rules_java`, `rules_shell`, `protobuf`) |

The initial generated configuration had an output scale near `2^80` at a
single approximately 50-bit modulus and missed the precision target. The
checked-in fixed profile gives the final output modulus room for the actual
generated scale; circuit semantics and test tolerance are unchanged.

Server reproducibility paths: `build/baseline/manifest.json`,
`build/baseline/lowered.mlir`, `build/baseline/model.go`,
`build/logs/reference-tests.log`, `build/logs/frontend-tests.log`,
`build/logs/compiler-tests.log`, and
`build/logs/heir-source-test-distdir.log`. The reference errors are empirical,
not an estimator result or a tail-probability certificate.

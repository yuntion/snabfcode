# Pass 2 encrypted regression

Run `bash src/tools/run_pass2.sh` with the prebuilt native and HEIR toolchains. Each run uses a new directory under `build/pass2` and keeps the source, marked IR, original CKKS and Go comparator, candidate/checked/lowered IR, native audit (frontier, selected DAG, decisions, routes and keys), corpus, and encrypted measurements.

## Precision and negative comparator

The preregistered numerical threshold is **absolute error at most `2^-12` (`0.000244140625`) on every logical output slot**, compared with the float64 dense-weight, true-SiLU oracle. It is unchanged for every arm. All arms run the same 64 source-derived inputs for their workload: 256 logical outputs for control and extended-profile coverage, and 4,096 for each width-64 network. MSE is reported for diagnosis; it cannot make an arm pass. Encryption uses fresh keys and randomness, so counts from distinct executions must not be combined.

| Arm | Any finite slot above the threshold | Nonfinite, shape, profile or ciphertext ABI mismatch |
| --- | --- | --- |
| `residual/heir` (original eager HEIR) | Report numerical **FAIL**, count every slot and continue the suite | Hard failure |
| Other stock HEIR comparators | Report **FAIL** after the complete comparison; fail the suite | Hard failure |
| Every SNABF arm, including residual and forced two-stage coverage | Report **FAIL** after the complete comparison; fail the suite | Hard failure |

The unchanged stock residual comparator already missed the threshold at trial 0, slot 1 in preflight `run.dJzeQ9Id` (absolute error about `4.74e-4`). That run stopped after one residual trial; it is not a 64-trial result. Do not substitute a different HEIR program or claim all comparators passed. The paired residual arms change only the public sink precision; the original stock client encrypts at L6 in the L7 profile, whereas planned clients use the full profile. The harness reads stock expectations from the original typed encryptor/evaluator returns, and planned expectations from the profile and audited sink. Runtime levels must match exactly; actual log scales may differ by at most 0.25 bits from nominal.

On November 19, 2026, an isolated replay of the **same** `run.dJzeQ9Id` residual programs and corpus completed all 64 trials per arm. Each `model.go` and the corpus were byte-compared with the preflight before launching. The raw observations and independent reductions are in `/home/fhecompiler/build/pass2-residual-probe.jOqmKX/unchanged-bpWsZyhF/residual/`:

| Residual arm | Numerical result | Slots above `2^-12` | Max absolute error | MSE | Input to output level | Actual output log scale |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Stock HEIR | **FAIL** | 1,089 / 4,096 | 1.388114373e-3 | 6.943165452e-8 | 6 to 0 | 26.012025454 |
| SNABF economy | **PASS** | 0 / 4,096 | 1.562991329e-4 | 9.820086786e-10 | 7 to 1 | 26 |
| SNABF precision | **PASS** | 0 / 4,096 | 1.357105274e-4 | 9.785182695e-10 | 7 to 0 | 26 |

These are observations from this replay, not reference constants or a claim about every encryption. The threshold itself was fixed before preflight. The change after the stock miss was to finish and report that comparison without suppressing measurements of the planned arms.

`trials.jsonl` contains exact promoted decrypted slots, ciphertext metadata and timings. `accuracy.json` contains the Go verdict, first miss, slot count, maximum absolute error and MSE over **all** slots. `metrics.json` independently recomputes and checks these fields from the raw trials and corpus. A numerical failure never discards later finite observations: the runner still exports `evidence.json`, then exits nonzero if **any hard gate** failed. Consult both `all_arms_meet_tolerance` and `regression_gates_passed`; only the latter may be true with `residual/heir` listed in `numerical_failures`. Incomplete or structurally invalid evidence is never certified.

The frozen-profile runs use the free planner. When they do not select the composed family, `composite_coverage` extends only Q by three specified primes on the same ring. Its `free` arm is a free-planner result; the explicitly filtered `forced_composite` and `forced_composite_tight` arms test two different two-stage programs. They are implementation coverage, not evidence of free selection or a security claim.

## Residual import contract

The source deliberately computes `h = W0*x; a = SiLU(h); y = W1*(h+a)`.
W0 uses the second reuse64 specification `{0,6,8}`, W1 the first `{0,7,9}`.
The fixed key price for both primary arms is 0.25 units/MiB, with no path cap.
The skip is the first matvec result, not an input added after decryption.
Stock HEIR with the reference profile and default greedy management rejects
the join (`ckks.add` operand scales 52 and 78). The **unchanged source**
verifies, lowers to Lattigo and translates to Go with the existing pipeline
option `greedy-modulus-switch-after-mul=true`. Use this option for both the
stock baseline and every planned arm of the residual workload.

Original stock CKKS probe (before native passes), observed on 2026-11-19:
`/home/fhecompiler/build/pass2-residual-probe.jOqmKX/residual64-ckks.mlir`,
`residual64-lattigo.mlir`, `residual64.go.txt`. SSA (level, nominal log scale):

```
%44 source (6,26)
%45 = W0(%44) (5,26)
%46 = kernel.eval_chebyshev(%45) (2,26)
%47 = ckks.level_reduce(%45, drop=2) (3,26)
%49 = ckks.mul_plain(%47, mask) (3,52)
%50 = ckks.rescale(%49) (2,26)
%51 = ckks.mul(%50, %46) (2,52)
%52 = ckks.relinearize(%51) (2,52)
%53 = ckks.rescale(%52) (1,26)             // SiLU
%54 = ckks.level_reduce(%45, drop=3) (2,26)
%56 = ckks.mul_plain(%54, mask) (2,52)
%57 = ckks.rescale(%56) (1,26)             // skip
%58 = ckks.add(%57, %53) (1,26)            // genuine two-ciphertext add
%61 = W1(%58) (0,26)
```

The backend join is `lattigo.ckks.add`. The actual runtime metadata must be
measured independently; the numbers above describe only HEIR's unmodified IR.

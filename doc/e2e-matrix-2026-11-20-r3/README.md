# Pass 1-4: planning and same-ciphertext emitter equivalence

Fresh encrypted evaluations; threshold 2^-12 at every encoded slot, including periodic replicas. Times are online Eval, not key setup, encryption, copy hashing or decoding.

## source

| Planning scenario | Live ciphertexts | Decision or refusal | N | L / log2(scale) | Conversion keys / bytes | Truth Linf / MSE | Misses / slots | Mean Eval ms |
|---|---:|---|---|---|---|---|---:|---:|
| reuse64-128 | 1 | `ring_down` | 16384 -> 8192 | 1 / 38.000000 | 1 / 3,146,080 | 1.0028582e-06 / 2.7079883e-14 | 0 / 262,144 | 157.00 |
| reuse64-512 | 1 | `switching_key_ext_modulus_security_rejected` | 16384 -> 16384 | 1 / 38.000000 | 0 / 0 | 9.949961e-07 / 2.7110058e-14 | 0 / 262,144 | 190.54 |
| cross-residual | 2 | `residual_bypass_crosses_cut_unconverted` | 16384 -> 16384 | 1 / 39.000000 | 0 / 0 | 9.9465346e-07 / 2.7069237e-14 | 0 / 262,144 | 191.74 |

| Fixed source ring | Inputs / timed Eval | N / L | Truth Linf direct / legacy | Paired Linf | Paired MSE | Truth misses direct / legacy | Paired misses |
|---|---|---|---|---:|---:|---|---:|
| control | 64 / 128 | 16384 / 1 | 3.278988e-05 / 3.297443e-05 | 2.306738e-07 | 2.3374115e-15 | 0 / 0 | 0 |
| strided64 | 64 / 128 | 16384 / 1 | 1.8418951e-06 / 1.8086677e-06 | 1.0882085e-07 | 1.5632611e-16 | 0 / 0 | 0 |
| reuse64 | 64 / 128 | 16384 / 1 | 9.9606746e-07 / 9.8184975e-07 | 9.1956781e-08 | 1.4264794e-16 | 0 / 0 | 0 |
| residual | 64 / 128 | 16384 / 1 | 2.9497076e-06 / 3.0484318e-06 | 1.1230154e-07 | 1.7100239e-16 | 0 / 0 | 0 |

256 encrypted inputs; 512 timed evaluations; 8 additional, untimed gadget probes. 1,048,576 slot pairs, maximum paired difference 2.30673795e-07, slot-weighted paired MSE 7.01846975e-16.

## pass3

| Planning scenario | Live ciphertexts | Decision or refusal | N | L / log2(scale) | Conversion keys / bytes | Truth Linf / MSE | Misses / slots | Mean Eval ms |
|---|---:|---|---|---|---|---|---:|---:|
| reuse64-128 | 1 | `ring_down` | 16384 -> 8192 | 1 / 38.000000 | 1 / 3,146,080 | 9.86212e-07 / 2.7078347e-14 | 0 / 262,144 | 156.16 |
| reuse64-512 | 1 | `switching_key_ext_modulus_security_rejected` | 16384 -> 16384 | 1 / 38.000000 | 0 / 0 | 9.9760873e-07 / 2.7107715e-14 | 0 / 262,144 | 190.02 |
| cross-residual | 2 | `residual_bypass_crosses_cut_unconverted` | 16384 -> 16384 | 1 / 39.000000 | 0 / 0 | 9.9030312e-07 / 2.707019e-14 | 0 / 262,144 | 192.18 |

| Fixed source ring | Inputs / timed Eval | N / L | Truth Linf direct / legacy | Paired Linf | Paired MSE | Truth misses direct / legacy | Paired misses |
|---|---|---|---|---:|---:|---|---:|
| control | 64 / 128 | 16384 / 1 | 3.2791992e-05 / 3.2974637e-05 | 2.3188835e-07 | 2.3373004e-15 | 0 / 0 | 0 |
| strided64 | 64 / 128 | 16384 / 1 | 1.8349003e-06 / 1.8126735e-06 | 9.2436716e-08 | 1.5699087e-16 | 0 / 0 | 0 |
| reuse64 | 64 / 128 | 16384 / 1 | 9.9990392e-07 / 9.7651722e-07 | 1.0612617e-07 | 1.4375615e-16 | 0 / 0 | 0 |
| residual | 64 / 128 | 16384 / 1 | 2.9473702e-06 / 3.0500838e-06 | 1.1401934e-07 | 1.7055139e-16 | 0 / 0 | 0 |

256 encrypted inputs; 512 timed evaluations; 8 additional, untimed gadget probes. 1,048,576 slot pairs, maximum paired difference 2.3188835e-07, slot-weighted paired MSE 7.0214969e-16.

## Interpretation and replay

`source` recompiles the supplied matrices against locked Pass 3 search bounds and concrete parameters. `pass3` is checkpoint replay, not another independent set of source inputs. The same corpus is used across modes.
Fixed-source-ring tests compare the native straight-line Go emitter with a freshly emitted historical HEIR/go-bind arm at the same upstream checkpoint. A single encryption is copied; both arms use the actual source evaluator and same-domain Decode. Mixed outputs are decrypted only in their own target domain.
Periodic replicas are not independent samples. A per-slot pass is neither a failure-probability proof nor a speedup claim. Lattice-estimator screens are conditional classical estimates; the 512-bit floor applies only to the conversion key.

Archive portability: Windows and Linux libm can differ in the last bit of exp. Each public exp(-h) binary64 witness was taken on the producer after an exact source/corpus replay. The offline checker independently bounds each witness against 90-digit Decimal.exp to one binary64 ulp, then reconstructs the entire frozen truth bit-for-bit. No tolerance is added to truth comparisons, decrypted slot errors, paired differences or the 2^-12 decision. Live source runs still use the original strict oracle. The four-gate negative record is in negative.json; raw Go, SSA, logs and slot pairs are in evidence.tar.gz.

The slot-weighted paired MSE uses a binary64 addition per workload in the fixed order control, strided64, reuse64, residual, matching the original measurement protocol independently of the Python version.

Archive-only verification (Python 3.10+, standard library; independent of HEIR, Go, the estimator and the build tree):

```bash
python3 audit/src/tools/archive_e2e_matrix.py check .
```

On the compiler server, from this archive directory, rerun in fresh directories:

```bash
python3 /home/fhecompiler/src/tools/e2e_matrix.py run --from source --inputs "$PWD/frozen" --out-root /home/fhecompiler/build/matrix-reproduce
python3 /home/fhecompiler/src/tools/e2e_matrix.py run --from pass3 --inputs "$PWD/frozen" --out-root /home/fhecompiler/build/matrix-reproduce
```

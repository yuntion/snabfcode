# Guarded periodic real diagonal, v1

This local theorem is about the **fresh DirectGo target**, not an attestation
that an archived executable used that target. It covers only a bound `mask`
call in standard-ring Lattigo v6.2.0, `[]float64`, batched `embedDouble`,
encoder precision 53, ordinary (not Montgomery) plaintext NTT. The checker
recomputes the use, key, ordered Q/P, level, payload bits, tiled argument and
physical scale. `E` below is `Scale.Float64()`, not the 128-bit metadata.
No public input to the proof is a measured ciphertext error or a claimed
noise distribution. Unsupported shapes and unsafe bounds do not release a site.

## Bound

Let `S` be the physical slots, `N` the ring dimension, `P` the shortest
bit-identical power-of-two tile, `k=log2(P)`, `W=max(abs(w_i))`. Here
`2<=P<=64`, `16<=S<=32768`, `2*S<=N<=65536`, `P|S`. Write `Q_l` for the
**product** of actual primes `q[0..level]`. Require `2^-300<=W<=2^60` and
`2^-300<=E<=2^300`. The following are exact rational upper bounds:

```
h = W * k * 2^-44
t = E * (W+h)                         require t < 2^48
U = floor(t + 9/16)                   require 2*U < Q_l
D = 2*P * (E*h + 9/16)
```

* In `SpecialiFFTDoubleUnrolled8` the initial `log2(S/P)` equal-half stages
  double the live block exactly and put zeros elsewhere. The remaining `k`
  stages use roots of order dividing `4P`. After division by `S`, bit reversal,
  real/imaginary packing and `rlwe.NTTSparseAndMontgomery`, only the `2P`
  indices spaced by `N/(2P)` may be nonzero.
* `epochs.RequireDiagonalEncodingV1` checks the **same deterministic**
  `GetRootsComplex128(2*N)` constructor as `ckks.NewEncoder`. All components
  must be finite, including off-grid multipliers of zero. Each on-grid
  component is checked within `2^-48` of its mathematical value. First-quadrant
  cosine intervals come from integer square roots of `(1+cos(2 theta))/2`,
  outward at 96 fractional bits, with exact signs/reflection. Comparison uses
  the intersection of the admissible neighborhoods of both endpoints. No
  host `math.Cos` ulp assertion, sampling or sheet of measured twiddles.
* In complex modulus, bound each ideal input of live stage `j` by
  `B=2^j*(S/P)*W`. Let `r` bound its error divided by `B`. The sum/difference
  of two approximate inputs has magnitude at most `2B*(1+r)` before rounding.
  A twiddle differs by less than `2^-47` in modulus. Round-to-nearest binary64
  has `u=2^-53` and absolute underflow error below `2^-1022` per real operation.
  Component sum/difference costs at most `u*2B*(1+r)+2*2^-1022` in modulus;
  four real products and two sums for a complex multiply cost less than
  `16u` times the product of the operand moduli plus `16*2^-1022` (the same
  envelope covers fusion). Even allowing `64u` in this stage, division by
  `2B` gives `1+r' <= (1+r)*(1+2^-45)`, since `B>=2^-300`. No overflow is
  possible in the stated range. The final division by the real power of two
  is exact except for absolute underflow. For `1<=k<=6`, these errors fit
  `h` after normalization. This bounds each component's departure from the
  exact special inverse transform; each ideal component has magnitude `<=W`.
* Under `t<2^48`, unsigned `SingleFloat64ToFixedPointCRT` truncates the
  scaled absolute value plus one half. Both binary64 roundings together
  cost at most `1/16`, including an addition crossing the power-of-two
  boundary. Thus integer rounding costs at most `9/16`. The signed integer
  coefficient is bounded by `U`, independent of whether it is positive.
  On each active limb `q<2^60`, the raw CRT word can be as large as
  `max(q,U)`. `MRedLazy(V,F)<2q` for a machine word and `F<q`, so every
  forward NTT addition fits if `max(q,U)+2*q*log2(2*S)<2^64`. This is
  checked per actual limb before crediting the polynomial. `2*U<Q_l` gives
  a unique centered lift even when `U` exceeds `q[0]/2`.
* A canonical slot evaluates the sparse polynomial on the unit circle; the
  triangle inequality gives `|sigma(C)_i-E*w_i|<=D` for all slots and their
  conjugates. The released site charges `A*ideal_abs*D` at the multiplication
  output. The `(A*E-output_scale)` ideal contribution was already charged.
  The `C*e` transformed parent is **not** a signed scalar. The separate
  `mul_plain` role stays open even when the convolution coefficient bound
  `(2P)*U*parent_coeff_abs` itself fits. No event probabilities are assigned.

## Compatibility and replay

Current emitted models with vector masks use the named guarded policy. A linked
old backend missing the symbol cannot satisfy it; a successful guarded `Bind`
records the checked ring in the runtime. Neither archived two-argument `Bind`
nor `RequireExactEncoding` grants this property. No IR seal is removed or
upgraded. The legacy engineering audit is not changed.

For a fresh target use `replay_noise_audit.py --independent --b2i
--guarded-diagonals --checked CHECKED --params PROGRAM --source-witness SOURCE
--basis BASIS --compare AUDIT`. Without `--guarded-diagonals`, independent
B2-i replay uses the previous DC-only contract: a new diagonal claim fails
comparison. The flag selects theorem semantics, not proof of a past run.
`test_b2i_noise.py --native OPT --go GO --fixtures FIXTURES --reports NEW_DIR`
also checks the live guard, all actual inverse-NTT limbs and all distinct
canonical embeddings using independent integer intervals. That probe is
corroboration, never an input to a probability certificate.

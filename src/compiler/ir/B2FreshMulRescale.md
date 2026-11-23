# Guarded DirectGo degree-one multiplication and rescale (v1)

This is an opt-in local proof, not a production noise certificate. Audit with
`snabf-opt --snabf-b2-mul-rescale '--snabf-check-plan=b2i-local=true noise-audit=out.json'`
on the sealed single-ring DirectGo input (add `require-noise-cert=true` to
the pass options to test denial). Also pass `--snabf-b2-mul-rescale` to
`--snabf-bind-lattigo=ring-go=model.go` after DirectGo lowering. Only that
newly generated `New()` returns the opaque `FreshModel`. It admits
`EvalFresh(x)` or `EvalFreshDecoded(x)`, not an external ciphertext,
mutable evaluator, plaintext or decryption oracle. The returned model can be
reused: neither repeated invocations nor rotations are independent draws.
The target is the pinned, unmodified generated v6.2.0 implementation. The flag
selects a theorem about this target; a seal alone is not binary attestation.
Old `Bind`, the guarded diagonal-only target, their audits and archives are
unchanged. No IR is resealed. No HEIR, native N1, mixed or external-key ABI
inherits this result.

## Premises and units

`B` bounds the maximum absolute integer coefficient of the **centered
decryption polynomial**, not of the ideal message or either ciphertext
component. `Q` is the product of the actual active ordered prefix including
q[0]. `sigma` below denotes the complex canonical embedding, not a standard
deviation. All bounds use integer or exact rational arithmetic. Instance
parameters, source scale, dynamic/public encode calls, operand indices,
width, slots, rotations and key IDs are reconstructed from the emitted plan.
The opaque generated handle checks finite real `|x|<=1`, width and the real
root table. It uses the same private runtime's `EncryptNew` with its own PK.
Bounds are uniform over this domain, **on every draw supported by the checked
samplers**; they give no conditional MGF, independence, or failure allocation.

The supported shape is a Standard ring, `1024<=N<=65536`, one P, at most
eight Q, all primes `<2^50`, Float64 encoder with default scale `E=2^d`,
`1<=d<47`, one prime per rescale. The runtime checks the concrete Xs
ternary `P=2/3,H=0`, Xe Gaussian `Sigma=3.2,Bound=19.2`, and generated
full-level singleton-P, base-two-zero Galois keys. The Gaussian sampler
rejects magnitudes above 19.2 *before* rounding to an integer, hence
`|e_j|<=19`; `|s_j|,|u_j|<=1`. These are support bounds, not standard
deviations. See `core/rlwe/{security,keygenerator,encryptor}.go` and
`ring/{sampler_gaussian,basis_extension}.go` in the pinned dependency.

* **Fresh input.** The checked real input tile has power-of-two width `w<=64`
  repeated across S slots. The guarded FFT argument in `B2iDiagonal.md`
  applies with the *envelope* W=1 (not the observed maximum): normalized
  coefficient error `h=log2(w)*2^-44`, quantization error `a=E*h+9/16`,
  coefficient bound `U=floor(E+a)`, slot encoding error `<=2*w*a`.
  Width one is the exact-DC butterfly. The word envelope
  `max(q_i,U)+2*q_i*log2(2*S)<2^64` follows from the above caps.
  For `encryptZeroPk`, write `pk0+pk1*s=e_pk mod QP` and center each of the
  two P-remainders of `u*pk + (e0,e1)` in the coefficient domain.
  With `H=(P-1)/2`, all coefficients of decrypted encryption noise have
  absolute value at most `T=floor((19*(2*N+1)+(N+1)*H)/P)`. Indeed the
  numerator is `u*e_pk+e0+e1*s-r0-r1*s`, divisible by P. The integral
  lift `B=U+T` is used only if `2*B<Q`. Slot error `<=2*w*a+N*T`.
* **Rotation support only.** In `DecomposeNTT/DecomposeAndSplit`, alpha=1
  gives one digit per active native limb, `|digit_j|<=q_i` (do not assume
  the symmetric half at the threshold). Among retained digits use triangle
  with the support 19 of **each** EVK row; do not add variances. For `K=1`
  the actual `gadget.project` subtracts `center_D(c1)`, it does not divide;
  `D=q[0]`, or `D=1` for `K=0`. The omitted term costs `N*(D-1)/2`.
  After the common P-ModDown and signed coefficient permutation one may add
  `J=N*(D-1)/2 + floor((19*N*sum(q[K:level+1])+(N+1)*H)/P)` to B and
  `N*J` to an available slot-error bound, provided `2*(B+J)<Q`.
  Negacyclic convolution, permutation of the ternary secret, shared hoists
  and shared EVKs preserve these triangle bounds. The EVK, permutation and
  truncation roles are **not** released by this support-only step. For one P,
  centered `ModUpPtoQ` has one normalized CRT digit in `[0,1)`: nonzero
  distance to its upper integer endpoint is at least `1/P`, well above
  binary64 conversion/division roundoff at `P<2^50`.
* **Nonuniform diagonal.** For the separately proved *real encoded*
  plaintext C with period `t`, coefficient bound Uc and slot error eta,
  `L=2*t*Uc` bounds its coefficient 1-norm; `M=min(L,Eplain*W+eta)`
  bounds `||sigma(C)||inf`. There is no reduction of the decrypted lift if
  `2*B*L<Q`. Let `e=sigma(v)-A*x`, with parent physical scale A and
  ideal slot envelope I. Use any propagated deterministic modulus envelope
  and `N*B+A*I`, taking the smaller. The complete transformed-parent term
  is bounded by `M*||e||inf`; it carries **no signed gain**. This includes
  the cross term between plaintext encoding error and ciphertext error.
  The encoding row already charges `A*I*eta` and the physical-scale ledger
  charges metadata drift. Neither is charged a second time.
* **One-prime rescale.** `schemes/ckks.Evaluator.Rescale` calls
  `DivRoundByLastModulusManyNTT(1,...)` on both degree-one ciphertext
  components. Let `p` be the actual dropped suffix, `Q'=Q/p`, `h=(p-1)/2`,
  and let r0,r1 be the two coefficientwise centered residues modulo p.
  The output decrypts to `(v-r0-r1*s)/p mod Q'`; it is an integral lift
  because `v` is the centered input decryption. Set `D=(N+1)*h` and
  `b=floor((B+D)/p)`. Only `2*b<Q'` proves **no** output recentering.
  The old parent error divides by exactly p; local deterministic slot error
  is `<=N*D/p`, not one half and not independent of the parent or the key.
  In the supported unrolled Standard kernel INTTLazy finishes with reducing
  `mulscalarmontgomeryvec` before the half-prime shift. Even if the lifted
  remainder is large on a small destination limb, every forward NTT word
  is bounded by `p+(3+2*log2(N))*q_i`, below 64 bits. The final MRed has a
  word operand, a reduced multiplier, and the original ciphertext limb is
  reduced. This covers the small q[0] without borrowing inactive primes.

Only the arithmetic support of aligned adds and encoded scalar products is
carried to the next operation. No unrelated legacy proof role is silently
cleared. Do not propagate this support through ciphertext-ciphertext mul,
relinearization, drop or other unsupported operations. An absent parent
gets an explicit blocker, not a half-Q success or an empirical replacement.

## Independent replay

`python3 src/tools/replay_noise_audit.py --independent --b2i --guarded-diagonals --mul-rescale --checked CHECKED --params PROGRAM --source-witness SOURCE --basis BASIS --compare AUDIT`

`python3 test/compiler/test_b2_mul_rescale.py --native OPT --go GO --fixtures FIXTURES --reports NEW_DIRECTORY`

The checker reports thresholds even on blocked production diagonals and
single-prime rescales. The Python implementation reconstructs these without
reading the proposed audit until comparison. Real Go probes include a
centered input that wraps on rescale, a legal same-key rounding witness,
active-limb product wrap on every real mask, sampler substitution and old
target rejection. They do not enter any formula. No sink bound, proof of
the common event, equation (15) margin or probability budget follows from a
local deterministic support calculation.

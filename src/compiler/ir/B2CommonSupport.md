# Same-key arithmetic support and the equation (15) dry-run

This is an opt-in audit of a **new guarded execution** of DirectGo
`epochs.RequireFreshMulRescaleV1/EvalFresh` (Lattigo v6.2.0). No inferred
law, fitted noise, archived executable, substituted ciphertext or decryption
sample enters a proof. See `B2FreshMulRescale.md` for fresh public-key
provenance, the concrete singleton-P digit bounds, NTT word ranges and the
source encoding envelope; `B2iDiagonal.md` for the actual Float64 periodic
encoder. A graph seal alone does not identify this runtime policy. The common
support flag adds an audit rule, not a backend change or a new probability
provider. Legacy reports are verified in their original mode, not upgraded.

## What is propagated

For every fixed permitted logical input and every key/sampler outcome in the
guarded support, let `v` be the integral, centered **decryption** (never an
individual ciphertext component). Carry degree, a bound B on every integer
coefficient, the all-slot ideal magnitude I in physical units, and an absolute
bound E on `sigma(v) - Delta*ideal`. Here sigma denotes evaluation at *all*
odd complex roots of `X^N+1`, not the probability parameter. Source and
public values have the actual repeated `slots` layout; all extra embeddings
of the sparse plaintext are copies or conjugates. The secret and errors need
not be sparse. Rotate permutes all roots. The degree-two value decrypts with
exactly the original s and s squared. No new key is sampled at joins, hoists
or successive products. `epochs/execute.go` makes PK, relin and Galois keys
from that same secret. `DirectGo.cpp` copies the operand of in-place Giant.

For an integer real-coefficient polynomial with coefficient sup norm B,
`|v(z)| <= (N/sqrt(2))*B`. Indeed take the real projection along the argument
of `v(z)`. The sum of squared direction cosines is N/2, since `sum_j z^(2j)=0`
at every odd root for this N. Cauchy-Schwarz gives the result. We round UP
with `kappa=46341/65536` (`2*46341^2 > 65536^2`); for two such factors their
product uses the exact factor 1/2. Conversely the inverse DFT gives
`max_j |v_j| <= max_z |v(z)|`. Use this only on the unreduced candidate whose
congruence is established by the operation, never on an already centered
residue to infer no prior wrap. Each integral candidate uses
`B' = floor(min(coefficient_route, I'+E'))`, with actual metadata drift in E'.
Require `B' < floor(Q_active/2)`. This slightly stronger test excludes even
the endpoint where v6.2.0 `ring.PolyToBigintCentered` maps the positive
floor to a negative integer. It is separate from the centered suffix used by
`DivRoundByLastModulusNTT` and prefix used by `gadget.project`. Unsupported
parents, key, degree, K or levels cannot create a new support receipt.

## Local identities

* An aligned sum uses the replayed integer multipliers. Triangle bounds
  apply without independence. Signed multipliers may combine **only for the
  same SSA value**. Negation and level drops preserve the polynomial subject
  to the destination check. Constant add/multiply use the already established
  signed encoded integer. For a nonuniform diagonal keep its encoded l1/slot
  bounds and encoding-at-output error; it never gets a signed scalar gain.
* Ciphertext multiply: the unreduced coefficient envelope is `N*B1*B2`.
  The complete slot error before metadata drift is
  `I1*E2 + I2*E1 + E1*E2`. An SSA square still pays the last term: a centered
  random sign squared equals one, not zero. This is a deterministic support
  bound for the nonlinear term, not a conditional MGF or a fitted mean.
* For a real rotate K=0/1 or relin K=1, let `P` be the *actual* sole auxiliary
  prime, `D=prod(q[:K])`, `h=(D-1)/2`, `J=sum(q[K:level+1])`. The input
  degree j is 1 for rotation, 2 for relinearization. Lattigo's expanded
  pure-RNS row encrypts its gadget message times P, with coefficient errors
  bounded by 19. Each native singleton digit has coefficient magnitude at
  most its own q_i, including the boundary. The two output components'
  centered P remainders cost at most `(N+1)*(P-1)/2` together after decrypt.
  The switch-only integer increment is therefore at most
  `T=floor((19*N*J+(N+1)*(P-1)/2)/P)`. The division is exact for the
  unreduced numerator, by the P congruence. This bound covers reuse and
  hoisting with no probabilistic decoupling of the digits, rows or remainders.
  The omitted projection is `-center_D(c_j)*s^j`; for rotation apply the
  signed automorphism to the result. This follows from `gadget/centered.go`
  adding the retained c0 (and c1 for relin) to the projected gadget product,
  and v6.2.0 `GenRelinearizationKey` generating the switch from s squared to s.
  The coefficient increment is `T+N^j*h`; the slot switch charge is
  `kappa*N*T` and separate cutoff charge is `N^2*h/2` (rotation) or
  `kappa*N^3*h/2` (relin). K=0 has no cutoff. No independence, centering in
  expectation, nor noise-free EVK is assumed. Unmodelled relin K=0/K>1,
  multi-P and mixed-ring inputs get no new rule.
* For the actual dropped suffix p, `r_i=center_p(c_i)` gives the integral
  degree-one decryption `(v-r0-r1*s)/p mod Q'`. Charge
  `(kappa*N+N^2/2)*(p-1)/(2*p)` in slots and carry the old error divided by p.
  Raw coefficient bound: `(B+(N+1)*(p-1)/2)/p`. The full same-key product
  appears in both bounds. Slot projection also improves the fresh source's
  `N*Tpk` to `kappa*N*Tpk`, retaining the original encoding envelope.

## Probability boundary and replay

For a fixed input in the declared domain we still need a common event G on
the actual joint execution, valid source encoding/encryption contracts on G,
and reference-approximation/decoding against the same sink. All reused key
and slot terms must be valid simultaneously. For conditional sub-Gaussian
terms combine identical signed source tokens first; for unknown dependence
use Hoelder, `V <= (sum_i |gain_i| sigma_i)^2`, not a sum of variances. A
probabilistic replacement for the nonlinear absolute product must prove the
two thresholds, bound the retained product on G, and pay the union of their
tail failures once. This implementation uses no truncation thresholds or
probability allocation for that product. Dropped gadget limbs are a different
(deterministically bounded) truncation.

`noise_eq15_dry_run.py` starts from source, ordered Q/P and checked SSA. It
evaluates the arithmetic-only stress `(sqrt(2)*E)/(Delta/4096)` and reports
single-charge counterfactuals; those ratios do not add in a nonlinear graph.
The zero-variance stress comparison itself uses `2*E^2 <= (Delta/4096)^2`
in rationals. Published displays using logarithms or square roots are not
issuance predicates. This is neither measured error nor a lower bound on
attainable accuracy. Full (15) needs `sqrt(2)*(b+sqrt(2*log(2*N/delta))*sigma)`
within tolerance AND `Pr[G^c]+sum(delta_sink)<=2^-40`. Actual b, V,
delta_valid, certified bound and margin are unknown, not zero.

For each graph, F is its independent fixture directory, OUT is a new local
audit file, OPT is this exact build. Explicit refusal is the expected exit 1.

```sh
"$OPT" --allow-unregistered-dialect --snabf-b2-mul-rescale --snabf-b2-common-support \
  "--snabf-check-plan=b2i-local=true require-noise-cert=true noise-audit=$OUT" "$F/checked.mlir" -o /dev/null
python3 src/tools/replay_noise_audit.py --independent --b2i --guarded-diagonals --mul-rescale --common-support \
  --checked "$F/checked.mlir" --params "$F/program.json" --source-witness "$F/source.mlir" \
  --basis "$F/basis.json" --baseline "$F/audit.json" --compare "$OUT"
python3 src/tools/noise_eq15_dry_run.py --common-support --fixture "$F" --compare "$OUT"
python3 test/compiler/test_b2_common_support.py --native "$OPT" --go "$GO" --fixtures "$FIXTURES" --reports "$NEW_DIR"
```

No CMake/driver change. Structural check without an audit request does not
run the noise checker. The Go regression reconstructs the projected secret
term independently and checks each centered EVK-only difference using one
freshly generated key per graph, every actual switch and level. This is
empirical corroboration, not a premise of any receipt.

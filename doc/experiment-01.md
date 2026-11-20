# E01: Joint budget decisions on an encrypted SiLU block

**2026-11-19:** [Pass 1 results](pass1-development.md) cover encrypted routing
on the 4x4 control and two 64x64 strided workloads, plus a separate bootstrap
decision. The next E01 integration is the Pass 2 SiLU frontier.

## Research question

Can selecting a linear routing schedule, polynomial evaluation DAG, RNS and
gadget settings, and a ring transition together produce a cheaper valid CKKS
program than selecting the same alternatives one region at a time? The
experiment tests that claim on one end-to-end workload before starting the
larger evaluation matrix in the paper.

## Workload and arms

The checked-in `Linear -> SiLU -> Linear` fixture is the integration control
for the numerical harness, with dyadic public weights, all input corners and
seeded random inputs. For the paper run, keep the same operator chain, grow
the strided linear layers to 64 by 64 or wider, and retain enough useful
post-cut linear work to expose whether conversion saves time. Add a residual
live across the cut, so lowering must reconcile all live operands. Use a
periodic packing case and a full-packed case with unequal values in trace
orbits. Exercise ring-switched bootstrap in a with-bootstrap fixture using
the same planner and backend, including a rejected transition.

Compare the fixed HEIR/Lattigo control, greedy per-region choices, and the
joint SN-ABF choice. Use the same public inputs and output tolerance;
regenerate cryptographic randomness independently. Run fixed-ring and
disable-truncation controls on the joint candidate space. For all comparisons
that claim security, use the same estimator settings and screen both arms.

## Implementation and acceptance order

1. **Register the dialect and intake.** Add TableGen types, ops and attributes
   for `!snabf.ct<shape,N,L,@profile>`, `snabf.lintrans`, `snabf.poly_eval`, and
   `snabf.convert`. Connect `src/frontend` to the verified HEIR/MLIR pipeline
   and build the SSA budget registry. Make negative verifier tests for
   layout, key, ring, scale and chain incompatibility. Distinguish level
   index `ell` from the number `ell+1` of active limbs. Keep the variance,
   canonical bias, failure-event allocations and security evidence distinct.
2. **Materialize Pass 1.** Connect the implemented subgroup tower and scoped
   Galois-key union selection to real packed layout extraction and emitted
   rotations. Log the inverse-rotated plaintext diagonals and compare the
   emitted routes against direct diagonals and the stock HEIR route. Use
   measured costs at the selected levels for the boot/convert comparison in
   Eq. (3), including repacking and downstream parameter costs.
3. **Materialize Pass 2.** Define the reference template before synthesizing
   its stage approximants. For SiLU use `u = tanh(x/4)`,
   `r(u) = u/(1+u*u)`, and the final DAG `x*(1/2+r(u))` with a bypass edge for
   `x`. The first two stages alone do not compute SiLU. For each transition
   certify local uniform residual on the ideal interval and a derivative
   bound on the hull of ideal/actual intervals, then propagate the error.
   Retain Pareto candidates with their actual evaluation DAGs and output
   interfaces. Align the paper's current scalar Bellman recurrence with this
   forward, template-indexed recurrence and Pareto output. Account for the
   bypass multiply, rescale and operand alignment explicitly.
4. **Materialize Pass 3.** Profile and implement the exact approximate
   CRT-gadget construction and RNS cancellation identity, then give HiGHS
   their costs, scale constraints, and bias/noise contracts. For `t` a
   conservative piecewise-linear upper envelope of `q^2` use all four
   inequalities for `y = b*t`, with `b` binary and finite bounds on `t`.
   The binary product is exact for the chosen envelope, not for `q^2`.
   Use a lower envelope for the RHS capacity. Instantiate distinct NTT primes
   and rerun alignment, output precision, wraparound and full public-key
   inventory checks on concrete `Q/P` and scales. The real CKKS program must
   exercise a nonzero admitted `k`; lowering to ordinary key switching while
   reporting an approximate-gadget choice does not count.
5. **Materialize Pass 4 and close the loop.** Use the tested Lattigo
   `ApplyEvaluationKey` projection and enforce the layout contract before
   emitting it. When full packing aliases distinct live values, use explicit
   independently verified repacking or reject. Run the converted suffix
   under its actual smaller-ring parameters and keys. Re-solve dependent
   candidates after changing a ring, layout or error budget. Check every
   public CKKS/RLWE instance, including actual conversion keys at the larger
   ring and extended key moduli, with a pinned estimator model. The target
   bit-security is a constraint, not an initial security result.

## Artifact and stop rule

For each run retain the input hash, fully lowered MLIR and Go, selected and
rejected plan records, actual `Q/P`, default and per-value scales, RNS levels,
live-operand layouts, exact gadget selections, public-key inventory and
estimator settings/results. Record runtime/serialization bytes, emitted
rotation and rescale counts, max error and MSE. Separate the plaintext
approximation residual, CKKS arithmetic error against that same polynomial
DAG, and full decoded error against SiLU. Use two independent data splits if
inputs or parameters are tuned, and report empirical accuracy separately
from the paper's conditional tail certificate.

E01 is ready for a paper result when the joint plan compiles the same workload
with real changed operations, executes encrypted end to end, satisfies the
same precision and security screens, and shows which coordinated choice
improved cost or avoided a failed configuration. One such result with the
controls above is enough to start writing the experimental section; the
full workload matrix is a separate milestone.

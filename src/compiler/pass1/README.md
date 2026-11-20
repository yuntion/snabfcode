# Native Pass 1

`snabf-pass1-candidates` publishes alternatives; `snabf-plan` chooses them on
the structured SSA graph. `snabf-check-plan` replays the physical inventory,
per-site shares, input budgets, and every rotation-path witness without solving
another MILP. Both lowering and backend binding check their inputs.

| Component | Responsibility |
| --- | --- |
| `SubgroupTower` | Lift packed offsets in the physical group, form the minimal subgroup, enumerate the complete index-two tower and occupied cosets. |
| `KeyReuseModel` | Price a fixed assignment. Per-site work is not de-duplicated; only storage is shared within an identical physical key family. |
| `../RouteSelection` | HiGHS one-hot selection with global key activation and path constraints. |
| `NativeRouting` | Read `snabf.lintrans` and profile/feedback, publish tower alternatives, context and charges. |
| `RingSwitch` | Equation (3), trace layout/level/variance feasibility and structured rejection reasons. No MLIR dependency. |
| `NativeBootstrap` | Bind the incoming SSA layout and level, replay decisions, and expose an executable standalone bootstrap only for the supported periodic profile. |
| [Shared Lattigo target](../Target/Lattigo/) | Emit model-specific Go directly from checked concrete plans, importing the pinned HEIR ordinary CKKS helpers. Pass 4 uses this by default; standalone concrete Pass 3 opts in with `--snabf-lower-ckks=direct-go`. The target's `go-bind emit` remains a historical comparator, not the new emitter. |
| `Planner.cpp` | Standalone C++ JSON inspection CLI (`snabf-pass1`). The native compiler and its regression runner do not require this CLI to plan or lower a module. |

`equivalent_splits` consists of `(s, stride, b_v, g_v)` tuples; `cosets` lists
only occupied representatives and their diagonals. A zero offset still denotes
a real diagonal, unlike empty support. `route-pricing=occupied` charges emitted
rotations; `route-pricing=dense_tower` instead uses the literal full-subgroup
work counts of Eq. (2). It evaluates all mathematical cuts individually.

Run `bash src/tools/run_pass1.sh` from the project root. CTest exercises the C++
decision kernel directly. Integration tests send actual HEIR/MLIR modules to
`snabf-opt`; Python drives the frontend and tests, not Pass 1 decisions. The
runner creates a new evidence directory per run; use `SNABF_PASS1_BUILD` for a
separate build tree under the project's `build/`.

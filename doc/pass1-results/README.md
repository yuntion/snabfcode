# Pass 1 server snapshot, 2026-11-19

`metrics.json` holds the nine encrypted workload/arm measurements. Each entry
in `workloads` supplies the input, frozen profile, and exact `plan.json` hashes.
`control/`, `strided64/`, `reuse64/`, and `boot/` each hold the corresponding
planner output. `pass1-run.log` is the full Pass 1 run, and `reference-tests.log`
is the separate regression run of the unchanged baseline and DownConv tests.
The bootstrap record uses normalized planning costs; it is not an encrypted
ring-switched-bootstrap measurement. See [the development report](../pass1-development.md).

#pragma once

#include "BoundedNoiseTransfer.h"
#include "pass2/AffineErrorForm.h"
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace snabf::ir::noise {
using pass2::AffineErrorForm;
using pass2::Exact;
inline constexpr double unset = std::numeric_limits<double>::quiet_NaN();
inline constexpr double theoremBudget = 0x1p-40;

struct Domain {
  std::string key_id;
  int ring_n = 0;
  std::vector<uint64_t> q, p; // actual, ordered, NEVER widths
  int rescale_primes = 1;
};
enum class Op { Source, Add, Sub, Neg, Drop, Rescale, ScalarMulPlain, AddPlain,
                CtMul, VectorMulPlain, Relin, Rotate, Down };
struct Node {
  std::string id;
  Op op;
  std::vector<size_t> inputs; // earlier SSA definitions; fan-out reuses one index
  size_t domain;             // result domain, including after a conversion
  int level, k;
  // Actual encoder metadata for Source or the encoded plaintext operand.
  // Never an output scale prediction, log2, or a planner's scale class.
  std::optional<Exact> encoded_scale;
  std::string input_draw, evk; // physical input draw (not key ID); physical EVK
  std::optional<double> real_scalar; // ideal signed scalar for ScalarMulPlain
  std::vector<double> plain_values; // complete payload for VectorMulPlain/AddPlain
  int64_t rotation = 0; // signed component permutation, never a scalar gain
  std::optional<Exact> effective_encoder_scale, source_bound;
  std::optional<BoundPlainCall> bound_plain_call; // opt-in checked DirectGo v1 only
  bool diagonal_mask = false; // actual EmitGraph Mask, never inferred from the SSA name
};
struct OutputPort { std::string id; size_t value; };
struct Topology {
  std::vector<Domain> domains;
  std::vector<Node> nodes;
  std::vector<OutputPort> outputs; // ALL returned ports, not selected witnesses
  int direct_go_slots = 0; // sealed emitter layout, not inferred from a plaintext receipt
  int fresh_width = 0; // nonzero ONLY for the explicit guarded EvalFresh target
  bool common_support = false; // additional audit for the same guarded target, not a law
};

enum class Role { Encode, Encrypt, Add, Sub, Neg, Drop, RescaleRound,
                  PlainEncode, PlainMul, PlainAdd, Product, Permute, Project,
                  KeySwitch, GadgetTrunc, DownConvRound, Output };
const char *name(Role);

enum class Evidence { Unverified, Empirical, ValidatedAnalytic };
struct CommonLaw {
  Evidence evidence = Evidence::Unverified;
  std::string event, proof_ref;
  // Remaining Pr[G^c] not itemized by the per-site contracts. An explicit
  // zero is an assumption/proof, not a default. Includes global wrap/domain.
  double global_failure = unset;
  // If nonempty, proof of JOINT independence of these actual input draws
  // conditional on precisely G, for each fixed plaintext and component.
  std::string joint_proof;
  std::set<std::string> independent_draws;
};
struct SiteBound {
  Evidence evidence = Evidence::Unverified;
  std::string event, formula_ref;
  double bias = unset, variance = unset; // canonical component units on G
  std::map<std::string, double> failures; // local events, each charged once
};
struct SiteQuery {
  const Topology &graph;
  const Node &node; // for Output: producer of the returned value
  Role role;
  const std::string &port_or_node_id;
  const CommonLaw &law;
  const Exact &physical_scale;
  const std::vector<std::optional<Exact>> &scales; // all physical SSA scales
  const std::vector<AffineErrorForm> &edges; // already replayed upstream
};
// Trust boundary IN CODE, not a reader for "validated" IR/profile flags.
// common/site must independently validate the materialized program, parameters,
// operator semantics, distributions, reachability and joint conditional law.
// Permute/Project/Product/vector PlainMul bound the COMPLETE transformed
// parent error (incl. cross terms); preserving a signed input identity there
// would permit invalid cancellation across slots. Other sites are incremental:
// rescale divides by the actual prime product; a real scalar uses the
// effective encoding scale. Ideal-message scale drift is charged separately.
// Encoding/FFT/rounding errors (including their cross terms) remain owed.
// PlainEncode is its contribution AT THE OP OUTPUT, not a raw plaintext bound.
// Output covers reference-function approximation and decoding, in canonical
// units. No built-in CKKS/empirical provider is installed.
struct AnalyticContracts {
  std::function<std::optional<CommonLaw>(const Topology &)> common;
  std::function<std::optional<SiteBound>(const SiteQuery &)> site;
};

enum class Mode { Engineering, Probability };
struct Options {
  Mode mode = Mode::Engineering;
  int precision_bits = 12;
  double target_failure_prob = theoremBudget;
  std::map<std::string, double> sink_delta;
  bool b2i = false;
};
struct SinkCertResult {
  std::string sink_id, physical_scale_exact;
  int ring_n = 0;
  // Scale and tolerance are lower, bias/variance/bound are upper. Disengaged
  // means unknown, not zero (in particular in an engineering-only audit).
  std::optional<double> actual_physical_scale, deterministic_bias_b,
      total_variance_proxy_v, allocated_delta_sink, required_tolerance_units,
      certified_bound_units;
  bool margin_passed = false;
};
struct SiteAudit { std::string id; Role role; std::string formula_ref; };
struct EdgeAudit {
  std::string id, key_id, evk, physical_scale_exact;
  int ring_n, level, k;
  std::string metadata_scale, effective_scale, ideal_bound, scale_bias;
  std::string join_branch, multiplier_a, multiplier_b, residual_a, residual_b;
};
struct CertAuditReport {
  Mode mode = Mode::Engineering;
  bool all_contracts_present = false, event_budget_passed = false,
       all_sinks_margin_passed = false, probabilistic_cert_issued = false;
  std::optional<double> total_delta_valid, sum_delta_sinks;
  double target_failure_prob = theoremBudget;
  std::string common_event, common_proof, joint_proof;
  std::vector<SinkCertResult> sinks;
  std::vector<SiteAudit> sites;
  std::vector<Domain> domains;
  std::vector<EdgeAudit> edge_inventory;
  // For each sink: merged_same_source_terms, conservative_holder_terms and
  // their exact signed coefficients/weighted sigmas; includes zero coefficients.
  std::map<std::string, AffineErrorForm> sink_forms;
  std::map<pass2::EventKey, Exact> validity_events;
  std::set<std::string> missing_contract_terms;
  bool b2i = false;
  std::vector<LocalTransfer> local_transfers;
  std::vector<std::string> errors;
  // Only the NOISE gate. Does not authorize emission without other gates.
  bool accepts() const {
    return errors.empty() && (mode == Mode::Engineering || probabilistic_cert_issued);
  }
};

// Independent of the planner, MLIR and any cached edge error record. Caller
// extracts the complete emitted topology and binds encoder metadata; never
// deserialize a Topology/AnalyticContracts pair from an unverified profile.
CertAuditReport verifyNoiseCertificate(const Topology &, const Options & = {},
                                      const AnalyticContracts * = nullptr);

} // namespace snabf::ir::noise

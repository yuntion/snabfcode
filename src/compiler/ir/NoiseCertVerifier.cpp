#include "NoiseCertVerifier.h"
#include "ExactScaleTracker.h"
#include "pass3/PrimeInstantiator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace snabf::ir::noise {
using namespace pass2;
namespace {
void check(bool ok, const std::string &reason) {
  if (!ok) throw std::invalid_argument(reason);
}

using Scale = std::optional<Exact>;

void checkDomain(const Domain &d) {
  check(!d.key_id.empty() && d.ring_n >= 4 &&
        std::has_single_bit(unsigned(d.ring_n)) && !d.q.empty() && !d.p.empty(),
        "invalid concrete domain: " + d.key_id);
  check(d.rescale_primes == 1 || d.rescale_primes == 2, "unsupported rescale mode");
  std::set<uint64_t> seen;
  for (const auto *chain : {&d.q, &d.p})
    for (auto p : *chain)
      check(p % (2ULL * d.ring_n) == 1 && seen.insert(p).second && pass3::prime(p),
            "invalid or repeated NTT prime: " + d.key_id);
}

// No edge scale, reported log2, gain, variance or margin is read from a plan.
struct Step {
  Scale ideal, bias;
  Exact ma = 1, mb = 1, gain = 1;
};

std::vector<Scale> replayPhysical(const Topology &g, CertAuditReport &r, std::vector<Step> &steps) {
  check(!g.nodes.empty() && !g.outputs.empty(), "empty graph or sink inventory");
  std::set<std::string> domains, ids, draws, ports;
  r.domains = g.domains;
  for (const auto &d : g.domains) {
    checkDomain(d);
    check(domains.insert(d.key_id).second, "duplicate key domain: " + d.key_id);
  }
  std::vector<bool> used(g.nodes.size());
  std::vector<Scale> scales;
  for (const auto &n : g.nodes) {
    auto need = [&](bool b, const char *why) { check(b, n.id + ": " + why); };
    need(!n.id.empty() && ids.insert(n.id).second, "empty or duplicate SSA site");
    need(n.domain < g.domains.size(), "unbound domain");
    const auto &d = g.domains[n.domain];
    need(n.level >= 0 && size_t(n.level) < d.q.size(), "unbound level");
    bool keyed = n.op == Op::Relin || n.op == Op::Rotate || n.op == Op::Down;
    bool plain = n.op == Op::ScalarMulPlain || n.op == Op::VectorMulPlain || n.op == Op::AddPlain;
    need(n.k >= 0 && n.k <= n.level && (keyed || n.k == 0), "invalid truncation k");
    need(keyed == !n.evk.empty(), "missing or unexpected physical EVK");
    need((n.op == Op::Source) == !n.input_draw.empty(), "missing or unexpected input draw");
    need(n.op == Op::Source || plain || !n.encoded_scale, "unexpected encoded scale");
    need(n.op == Op::Source || plain || !n.effective_encoder_scale, "unexpected effective scale");
    if ((n.op == Op::Source || plain) && (!n.encoded_scale || !n.effective_encoder_scale))
      r.missing_contract_terms.insert("node/" + n.id + "/missing_encoding_scale_binding");
    if (n.effective_encoder_scale)
      need(*n.effective_encoder_scale == roundCKKSScale(*n.effective_encoder_scale), "invalid effective scale");
    if (n.source_bound) need(n.op == Op::Source && *n.source_bound >= 0, "invalid source envelope");
    if (n.encoded_scale)
      need(*n.encoded_scale == roundCKKSScale(*n.encoded_scale), "inexact scale literal");
    need((n.op == Op::ScalarMulPlain) == n.real_scalar.has_value() &&
         (!n.real_scalar || std::isfinite(*n.real_scalar)), "invalid signed plaintext scalar");
    need((n.op == Op::VectorMulPlain || n.op == Op::AddPlain) == !n.plain_values.empty() &&
         std::all_of(n.plain_values.begin(), n.plain_values.end(), [](double x) { return std::isfinite(x); }),
         "missing or invalid plaintext payload");
    need((n.op == Op::Rotate) == (n.rotation != 0) && n.rotation > -int64_t(d.ring_n / 2) &&
         n.rotation < d.ring_n / 2, "invalid component permutation");
    size_t arity = n.op == Op::Source ? 0u :
        n.op == Op::Add || n.op == Op::Sub || n.op == Op::CtMul ? 2u : 1u;
    need(n.inputs.size() == arity, "wrong primitive arity");
    for (size_t a : n.inputs) {
      need(a < scales.size(), "non-topological operand");
      const Node &in = g.nodes[a];
      used[a] = true;
      if (n.op == Op::Down) {
        const auto &src = g.domains[in.domain];
        need(in.domain != n.domain && d.ring_n < src.ring_n && src.ring_n % d.ring_n == 0 &&
             d.q.size() <= src.q.size() && std::equal(d.q.begin(), d.q.end(), src.q.begin()) &&
             d.p == src.p, "incompatible ring-down basis");
      } else need(in.domain == n.domain, "implicit change of ring or key");
      need(n.op == Op::Drop ? in.level >= n.level :
           n.op == Op::Rescale ? in.level - d.rescale_primes == n.level : in.level == n.level,
           "physical level mismatch");
    }
    Scale s = arity ? scales[n.inputs[0]] : Scale{};
    Step step;
    step.ideal = arity ? steps[n.inputs[0]].ideal : n.source_bound;
    Scale plainBound;
    if (plain) {
      Exact b = 0;
      if (n.real_scalar) b = exactBinary(std::abs(*n.real_scalar));
      for (double x : n.plain_values) b = std::max(b, exactBinary(std::abs(x)));
      plainBound = b;
    }
    std::string branch;
    Exact da = 0, db = 0;
    Scale ba = step.ideal, bb;
    const auto meta = n.effective_encoder_scale ? n.encoded_scale : Scale{};
    switch (n.op) {
    case Op::Source:
      need(draws.insert(n.input_draw).second, "duplicate definition of one physical input; reuse its SSA value");
      s = meta;
      if (s) da = *n.effective_encoder_scale - *s;
      break;
    case Op::ScalarMulPlain: case Op::VectorMulPlain: {
      if (step.ideal) *step.ideal *= *plainBound;
      ba = step.ideal;
      if (s && meta) {
        // v1 charged upstream drift on its defining edge. Its input reference
        // here is metadata, not a claim of a homogeneous effective gain.
        auto p = ExactScaleTracker::multiply({*s, *s}, {*meta, *n.effective_encoder_scale});
        s = p.metadata;
        da = p.effective - *s;
        if (n.real_scalar) step.gain = *n.effective_encoder_scale * exactBinary(*n.real_scalar);
      } else s.reset();
      break;
    }
    case Op::CtMul: {
      const auto &b = scales[n.inputs[1]];
      auto &other = steps[n.inputs[1]].ideal;
      if (step.ideal && other) *step.ideal *= *other; else step.ideal.reset();
      ba = step.ideal;
      if (s && b) {
        auto p = ExactScaleTracker::multiply({*s, *s}, {*b, *b});
        s = p.metadata;
        da = p.effective - *s;
      }
      else s.reset();
      break;
    }
    case Op::Rescale: {
      std::optional<ScaleState> ref;
      if (s) ref = ScaleState{*s, *s};
      for (int l = g.nodes[n.inputs[0]].level; l > n.level; --l) {
        if (ref) ref = ExactScaleTracker::replayPrimeDivision(*ref, d.q, l);
        else step.gain /= Exact(std::to_string(d.q[l])); // known even without input metadata
      }
      if (ref) {
        step.gain = ref->effective / *s;
        s = ref->metadata;
        da = ref->effective - *s;
      }
      break;
    }
    case Op::Add: case Op::Sub: case Op::AddPlain: {
      const auto &b = n.op == Op::AddPlain ? meta : scales[n.inputs[1]];
      bb = n.op == Op::AddPlain ? plainBound : steps[n.inputs[1]].ideal;
      if (step.ideal && bb) *step.ideal += *bb; else step.ideal.reset();
      if (s && b) {
        branch = *s == *b ? "equal" : *s < *b ? "multiply_left" : "multiply_right";
        const auto op = n.op == Op::Sub ? JoinOp::Sub : JoinOp::Add;
        // v1 projects just the two residual coefficients. Its exact-only bias
        // and missing-bound convention below are part of the frozen audit ABI.
        try {
          auto join = ExactScaleTracker::analyzeAddSubMismatch(op, {*s, *s},
              {*b, n.op == Op::AddPlain ? *n.effective_encoder_scale : *b}, {}, {});
          step.ma = mpz_class(std::to_string(join.lhs_int_mul));
          step.mb = mpz_class(std::to_string(join.rhs_int_mul));
          s = join.out_meta_scale;
          da = join.lhs_signed_effect;
          db = op == JoinOp::Sub ? -join.rhs_signed_effect : join.rhs_signed_effect; // pre-subtraction v1 coefficient
        } catch (const UnsupportedAddAlignment &e) {
          throw std::invalid_argument(n.id + ": " + e.what());
        }
      }
      else s.reset();
      break;
    }
    case Op::Down: step.ideal.reset(); break; // no certified projection envelope here
    case Op::Neg: case Op::Drop: case Op::Relin: case Op::Rotate: break;
    default: throw std::invalid_argument(n.id + ": unknown opcode");
    }
    if (!s) r.missing_contract_terms.insert("node/" + n.id + "/physical_scale");
    else if ((da == 0 || ba) && (db == 0 || bb))
      step.bias = Exact(abs(da) * ba.value_or(0) + abs(db) * bb.value_or(0));
    else r.missing_contract_terms.insert("node/" + n.id + (branch.empty() ? "/scale_drift_message_bound" : "/unequal_physical_join_transfer"));
    auto txt = [](const Scale &v) { return v ? v->get_str() : std::string{}; };
    r.edge_inventory.push_back({n.id, d.key_id, n.evk, txt(s), d.ring_n, n.level, n.k,
        txt(meta), txt(n.effective_encoder_scale), txt(step.ideal), txt(step.bias), branch,
        branch.empty() ? "" : step.ma.get_str(), branch.empty() ? "" : step.mb.get_str(),
        s ? da.get_str() : "", s ? db.get_str() : ""});
    steps.push_back(std::move(step));
    scales.push_back(std::move(s));
  }
  for (const auto &p : g.outputs) {
    check(!p.id.empty() && ports.insert(p.id).second && p.value < g.nodes.size(),
          "duplicate or unbound output: " + p.id);
    used[p.value] = true;
  }
  for (size_t i = 0; i < used.size(); ++i)
    check(used[i], "terminal SSA value absent from sink inventory: " + g.nodes[i].id);
  return scales;
}

DependentKind category(Role role) {
  switch (role) {
  case Role::RescaleRound: return DependentKind::RescaleRound;
  case Role::GadgetTrunc: return DependentKind::GadgetTrunc;
  case Role::KeySwitch: return DependentKind::SharedEvkSwitch;
  case Role::DownConvRound: return DependentKind::DownConvRound;
  default: return DependentKind::UnknownDependent;
  }
}

std::vector<Role> obligations(const Node &n) {
  using R = Role;
  std::vector<Role> roles;
  switch (n.op) {
  case Op::Source: return {R::Encode, R::Encrypt};
  case Op::Add: return {R::Add};
  case Op::Sub: return {R::Sub};
  case Op::Neg: return {R::Neg};
  case Op::Drop: return {R::Drop};
  case Op::Rescale: return {R::RescaleRound};
  case Op::ScalarMulPlain: case Op::VectorMulPlain: return {R::PlainEncode, R::PlainMul};
  case Op::AddPlain: return {R::PlainEncode, R::PlainAdd};
  case Op::CtMul: return {R::Product};
  case Op::Relin: roles = {R::KeySwitch}; break;
  case Op::Rotate: roles = {R::Permute, R::KeySwitch}; break;
  case Op::Down: roles = {R::Project, R::KeySwitch, R::DownConvRound}; break;
  default: throw std::invalid_argument(n.id + ": unknown opcode");
  }
  if (n.k) roles.push_back(R::GadgetTrunc);
  return roles;
}
} // namespace

const char *name(Role r) {
  switch (r) {
  case Role::Encode: return "encode";
  case Role::Encrypt: return "encrypt";
  case Role::Add: return "add";
  case Role::Sub: return "sub";
  case Role::Neg: return "neg";
  case Role::Drop: return "level_reduce";
  case Role::RescaleRound: return "rescale_round";
  case Role::PlainEncode: return "plain_encoding_at_output";
  case Role::PlainMul: return "mul_plain";
  case Role::PlainAdd: return "add_plain";
  case Role::Product: return "ct_mul_including_cross_term";
  case Role::Permute: return "permute_input_components";
  case Role::Project: return "ring_project_input_components";
  case Role::KeySwitch: return "evk_switch";
  case Role::GadgetTrunc: return "gadget_trunc";
  case Role::DownConvRound: return "downconv_round";
  case Role::Output: return "reference_approx_and_decode";
  }
  throw std::invalid_argument("unknown primitive role");
}

namespace {
void audit(const Topology &g, const Options &o, const AnalyticContracts *provider,
           CertAuditReport &r) {
  check(o.mode == Mode::Engineering || o.mode == Mode::Probability, "invalid noise mode");
  check(o.precision_bits >= 0 && o.precision_bits <= 16384, "invalid precision bits");
  Exact target = exactBinary(o.target_failure_prob);
  check(target > 0 && target <= exactBinary(theoremBudget), "target exceeds Theorem 1 or is not positive");
  r.b2i = o.b2i;
  check(!o.b2i || !provider, "cannot mix B2-i gains with a legacy conditional provider");
  std::vector<Step> steps;
  auto scales = replayPhysical(g, r, steps);
  std::set<std::string> outputs;
  for (const auto &s : g.outputs) outputs.insert(s.id);
  for (const auto &[id, p] : o.sink_delta)
    check(outputs.contains(id) && std::isfinite(p) && p > 0 && p < 1,
          "orphan or invalid sink allocation: " + id);

  auto law = provider && provider->common ? provider->common(g) : std::nullopt;
  AffineErrorForm global;
  if (!law || law->evidence != Evidence::ValidatedAnalytic) {
    law.reset();
    r.missing_contract_terms.insert("common/validated_joint_law_and_domain");
  } else {
    check(!law->event.empty() && !law->proof_ref.empty(), "anonymous common law");
    global.addEvent({"common", "remaining_validity_failure"}, exactBinary(law->global_failure));
    r.common_event = law->event;
    r.common_proof = law->proof_ref;
    r.joint_proof = law->joint_proof;
    check(law->joint_proof.empty() == law->independent_draws.empty(), "incomplete joint independence proof");
    std::set<std::string> draws;
    for (const auto &n : g.nodes) if (n.op == Op::Source) draws.insert(n.input_draw);
    for (const auto &id : law->independent_draws)
      check(draws.contains(id), "independence proof references absent draw: " + id);
  }

  std::vector<AffineErrorForm> edges;
  edges.reserve(g.nodes.size());
  std::vector<LocalStep> local;
  auto bind = [&](AffineErrorForm &f, size_t i, Role role, const std::string &id) {
    const auto &n = g.nodes[i];
    std::string key = (role == Role::Output ? "sink/" : "node/") + id + "/" + name(role);
    auto &row = r.sites.emplace_back(SiteAudit{key, role, {}});
    if (o.b2i && role != Role::Output) {
      for (const auto &t : local.at(i).rows) {
        if (t.site != key) continue;
        r.local_transfers.push_back(t);
        if (!t.formula.empty()) {
          row.formula_ref = t.formula;
          f.addBias(t.bias.value());
          return; // deterministic local identity; NOT a CommonLaw or an event
        }
      }
    }
    std::optional<SiteBound> b;
    if (law && scales[i] && provider->site)
      b = provider->site({g, n, role, id, *law, *scales[i], scales, edges});
    if (!b || b->evidence != Evidence::ValidatedAnalytic) {
      f.markMissing(key);
      r.missing_contract_terms.insert(key);
      return;
    }
    check(b->event == law->event && !b->formula_ref.empty(), key + ": not proved on the common event");
    check(std::isfinite(b->bias) && b->bias >= 0 && std::isfinite(b->variance) && b->variance >= 0,
          key + ": invalid canonical bound");
    row.formula_ref = b->formula_ref;
    f.addBias(exactBinary(b->bias));
    for (const auto &[reason, p] : b->failures) {
      EventKey e{key, reason};
      f.addEvent(e, exactBinary(p));
      global.addEvent(e, exactBinary(p));
    }
    Exact v = exactBinary(b->variance);
    if (role == Role::Encrypt) {
      auto proof = law->independent_draws.contains(n.input_draw) ? law->joint_proof : "";
      f.addLinear(n.input_draw, {1, v, NoiseSourceKind::FreshEncrypt, b->formula_ref, proof});
    } else if (v != 0) {
      f.addDependent({key, n.evk, category(role), sqrtUpper(v), b->formula_ref});
    }
  };

  for (size_t i = 0; i < g.nodes.size(); ++i) {
    const auto &n = g.nodes[i];
    if (o.b2i) {
      local.push_back(boundedStep(g, i, local, n.inputs.empty() ? Scale{} : scales[n.inputs[0]],
          n.inputs.empty() ? Scale{} : steps[n.inputs[0]].ideal, bool(scales[i]), steps[i].ma, steps[i].mb));
      if (n.op == Op::ScalarMulPlain && !local.back().rows.front().formula.empty())
        steps[i].gain = local.back().rows.front().lhs_gain.value();
    }
    AffineErrorForm f;
    for (size_t a : n.inputs) f.inheritObligations(edges[a]);
    if (!scales[i]) f.markMissing("node/" + n.id + "/physical_scale");
    auto unequal = "node/" + n.id + "/unequal_physical_join_transfer";
    if (r.missing_contract_terms.contains(unequal)) f.markMissing(unequal);
    if (steps[i].bias) f.addBias(*steps[i].bias);
    else f.markMissing("node/" + n.id + "/scale_transfer");
    switch (n.op) {
    case Op::Source: case Op::CtMul: case Op::Rotate: case Op::Down: case Op::VectorMulPlain:
      break; // complete transformed-input bound required; no guessed linear gain
    case Op::Add: case Op::Sub:
      if (n.op == Op::Sub && n.inputs[0] == n.inputs[1]) {
        // Identical SSA components cancel even an otherwise untracked bias
        // or residual. The defining and subtraction sites still owe proofs.
        f.addScaled(edges[n.inputs[0]], 0);
      } else {
        f.addScaled(edges[n.inputs[0]], steps[i].ma);
        f.addScaled(edges[n.inputs[1]], n.op == Op::Sub ? -steps[i].mb : steps[i].mb);
      }
      break;
    case Op::ScalarMulPlain: case Op::Rescale:
      if (scales[i]) {
        f.addScaled(edges[n.inputs[0]], steps[i].gain);
      }
      break;
    case Op::AddPlain:
      f.addScaled(edges[n.inputs[0]], steps[i].ma); break;
    case Op::Neg: case Op::Drop: case Op::Relin:
      f.addScaled(edges[n.inputs[0]], n.op == Op::Neg ? -1 : 1);
      break;
    }
    for (Role role : obligations(n)) bind(f, i, role, n.id);
    edges.push_back(std::move(f));
  }

  Exact tails = 0;
  const Exact precision(mpz_class(1), mpz_class(1) << o.precision_bits);
  for (const auto &p : g.outputs) {
    auto f = edges[p.value];
    bind(f, p.value, Role::Output, p.id);
    auto &s = r.sinks.emplace_back();
    s.sink_id = p.id;
    s.ring_n = g.domains[g.nodes[p.value].domain].ring_n;
    if (const auto &scale = scales[p.value]) {
      s.physical_scale_exact = scale->get_str();
      s.actual_physical_scale = lowerDouble(*scale);
      s.required_tolerance_units = lowerDouble(*scale * precision);
    }
    if (auto it = o.sink_delta.find(p.id); it != o.sink_delta.end()) {
      s.allocated_delta_sink = it->second;
      tails += exactBinary(it->second);
    } else r.missing_contract_terms.insert("sink/" + p.id + "/tail_allocation");
    if (law && f.missingContracts().empty() && scales[p.value]) {
      const Exact v = f.varianceUpper();
      s.deterministic_bias_b = upperDouble(f.bias());
      s.total_variance_proxy_v = upperDouble(v);
      if (s.allocated_delta_sink) {
        Exact logTerm = logUpper(Exact(2) * s.ring_n / exactBinary(*s.allocated_delta_sink));
        Exact upper = sqrtUpper(2) * (f.bias() + sqrtUpper(2 * v * logTerm));
        s.certified_bound_units = upperDouble(upper);
        s.margin_passed = *s.certified_bound_units <= *s.required_tolerance_units;
      }
    }
    r.sink_forms.emplace(p.id, std::move(f));
  }

  r.validity_events = global.events();
  const bool allTails = o.sink_delta.size() == g.outputs.size();
  if (allTails) r.sum_delta_sinks = upperDouble(tails);
  // Unknown local validity failures are not displayed as a total of zero.
  bool allSites = law && std::all_of(r.sites.begin(), r.sites.end(),
                                     [](const auto &s) { return !s.formula_ref.empty(); });
  if (allSites) r.total_delta_valid = upperDouble(global.deltaValid());
  r.all_contracts_present = allSites && r.missing_contract_terms.empty();
  r.event_budget_passed = allSites && allTails && global.deltaValid() + tails <= target;
  r.all_sinks_margin_passed = !r.sinks.empty() && std::all_of(r.sinks.begin(), r.sinks.end(),
                                                            [](const auto &s) { return s.margin_passed; });
  r.probabilistic_cert_issued = o.mode == Mode::Probability && r.all_contracts_present &&
      r.event_budget_passed && r.all_sinks_margin_passed;
}
} // namespace

CertAuditReport verifyNoiseCertificate(const Topology &g, const Options &o, const AnalyticContracts *p) {
  CertAuditReport r;
  r.mode = o.mode;
  r.target_failure_prob = o.target_failure_prob;
  try { audit(g, o, p, r); }
  catch (const std::exception &e) { r.errors.emplace_back(e.what()); }
  return r;
}
} // namespace snabf::ir::noise

#include "ir/NoiseCertVerifier.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace snabf::ir::noise;
using namespace snabf::pass2;
void expect(bool test, const char *why) { if (!test) throw std::runtime_error(why); }
template<class F> void rejects(F fn, const char *why) {
  try { fn(); } catch (const std::exception &) { return; }
  throw std::runtime_error(why);
}

Node node(std::string id, Op op, std::vector<size_t> args = {}, int level = 1) {
  Node n{};
  n.id = std::move(id); n.op = op; n.inputs = std::move(args); n.level = level;
  return n;
}
Node source(std::string id, std::string draw, uint64_t scale = 65536) {
  auto n = node(std::move(id), Op::Source);
  n.input_draw = std::move(draw);
  n.encoded_scale = Exact(std::to_string(scale));
  n.effective_encoder_scale = n.encoded_scale;
  return n;
}
Topology fixture() {
  // Tiny concrete arithmetic domain, deliberately NOT a secure deployment.
  // The noise gate does not replace the independent security/emission gates.
  return {{{"K", 8, {97, 193}, {241}}},
          {source("u", "draw-A"), source("w", "draw-B"), node("sum", Op::Add, {0, 1})},
          {{"out", 2}}};
}
Options certify(const Topology &g) {
  Options o;
  o.mode = Mode::Probability; o.precision_bits = 4;
  for (const auto &s : g.outputs) o.sink_delta.emplace(s.id, 0x1p-44);
  return o;
}
// Test-only mathematical hypothesis: on G each fixed component of the two
// incoming draws is an independent centered Rademacher (+/-1). The encode,
// addition and decode residuals vanish, and G has probability one. These
// formula fixtures are NOT claims about an RLWE implementation or a sample.
using Bound = std::optional<SiteBound>;
SiteBound bound(double b = 0, double v = 0) {
  return {Evidence::ValidatedAnalytic, "G", "fixture/component-lemma", b, v, {}};
}
AnalyticContracts proofs() {
  return {[](const Topology &) -> std::optional<CommonLaw> {
            return CommonLaw{Evidence::ValidatedAnalytic, "G", "fixture/G", 0,
                             "fixture/joint-product-on-G", {"draw-A", "draw-B"}};
          }, [](const SiteQuery &q) -> Bound { return bound(0, q.role == Role::Encrypt ? 1 : 0); }};
}
void noGroup(AnalyticContracts &p) {
  p.common = [](const Topology &) -> std::optional<CommonLaw> {
    return CommonLaw{Evidence::ValidatedAnalytic, "G", "fixture/G", 0, {}, {}};
  };
}

void affineArithmetic() {
  AffineErrorForm a;
  a.addLinear("rho", {1, 3, NoiseSourceKind::ProvenIndependent, "agg", "joint"});
  a.addEvent({"r", "valid"}, exactBinary(0x1p-43));
  auto twice = a;
  twice.addScaled(a, 1);
  expect(twice.varianceUpper() == 12 && twice.linearTerms().at("rho").signed_coeff == 2,
         "previously aggregated independent innovation lost its identity");
  twice.addScaled(twice, -1);
  expect(twice.varianceUpper() == 0 && twice.deltaValid() == a.deltaValid(), "alias cancellation/validity");
  a.addLinear("other", {1, 4, NoiseSourceKind::FreshEncrypt, "new", "joint"});
  expect(a.varianceUpper() == 7, "joint product on G not used");
  a.addDependent({"k0", "same-evk", DependentKind::SharedEvkSwitch, 2, "ks"});
  a.addDependent({"k1", "same-evk", DependentKind::SharedEvkSwitch, 3, "ks"});
  expect(a.computeTotalVarianceProxy() > 58 && a.computeTotalVarianceProxy() < 59, "independent/Holder boundary");
  rejects([&] { a.addLinear("rho", {1, 9, NoiseSourceKind::ProvenIndependent, "agg", "joint"}); }, "conflicting source accepted");
  rejects([&] { a.addLinear("bad", {1, 1, NoiseSourceKind::FreshEncode, "enc", "joint"}); }, "encoding marked independent");
  rejects([&] { a.addEvent({"r", "valid"}, 0); }, "conflicting same-event budgets");

  // Binary64 flattens (2^60 + 1) - 2^60. The true signed coefficient is 1.
  AffineErrorForm tiny;
  Exact big(mpz_class(1) << 60);
  for (const Exact &c : std::vector<Exact>{big, 1, -big})
    tiny.addLinear("u", {c, 1, NoiseSourceKind::FreshEncrypt, "same", ""});
  expect(tiny.varianceUpper() == 1, "signed coefficient rounded through zero");
  AffineErrorForm noProof;
  noProof.addLinear("alpha", {1, 4, NoiseSourceKind::FreshEncrypt, "src", ""});
  noProof.addLinear("beta", {-1, 9, NoiseSourceKind::FreshEncrypt, "src", ""});
  expect(noProof.varianceUpper() == 25, "different source labels were a proof of independence");
  Exact sub = exactBinary(std::numeric_limits<double>::denorm_min()) / 3;
  expect(upperDouble(sub) > 0 && lowerDouble(sub) == 0, "outward subnormal conversion");
}

void jointAndFanout() {
  auto g = fixture(); auto o = certify(g); auto p = proofs();
  auto r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.accepts() && r.sinks[0].total_variance_proxy_v == 2,
         "conditional Rademacher certificate");
  auto eng = o; eng.mode = Mode::Engineering;
  r = verifyNoiseCertificate(g, eng, &p);
  expect(r.accepts() && r.all_contracts_present && !r.probabilistic_cert_issued, "engineering issued a certificate");
  r = verifyNoiseCertificate(g); // no provider, no allocations
  expect(r.accepts() && !r.probabilistic_cert_issued && !r.sinks[0].total_variance_proxy_v &&
         r.missing_contract_terms.contains("node/u/encrypt"), "engineering treated missing as zero");
  noGroup(p);
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.sinks[0].total_variance_proxy_v == 4, "distinct inputs silently assumed independent");
  g.nodes[2].inputs = {0, 0};
  g.outputs.push_back({"unused_by_sum", 1}); o = certify(g); p = proofs();
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].total_variance_proxy_v == 4, "u+u incorrectly uses 2v");
  g.nodes[2].op = Op::Sub;
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].total_variance_proxy_v == 0, "u-u failed to cancel");
  p.site = [](const SiteQuery &s) -> Bound {
    auto b = s.role == Role::Encode ? bound(3, 7) : s.role == Role::Sub ? bound(1, 2) : bound();
    if (s.role == Role::Encode && s.node.id == "u") b.failures["wrap"] = 0x1p-45;
    return b;
  };
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].deterministic_bias_b == 1 &&
         *r.sinks[0].total_variance_proxy_v >= 2 && *r.sinks[0].total_variance_proxy_v < 2.000001 &&
         r.total_delta_valid == 0x1p-45, "identical SSA subtraction discarded local residual or global obligation");
  g.nodes[1].input_draw = "draw-A";
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "two independent definitions of one physical draw");
}

void budgetsAndEvidence() {
  auto g = fixture(); g.outputs.push_back({"tap", 0});
  auto o = certify(g); auto p = proofs();
  p.site = [](const SiteQuery &q) -> Bound {
    auto b = bound();
    if (q.role == Role::Encode) b.failures["wrap"] = 0x1p-43;
    return b;
  };
  auto r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.total_delta_valid == 0x1p-42 && r.sum_delta_sinks == 0x1p-43,
         "common event lost, or charged again at fanout/second sink");
  o.sink_delta["out"] = 0x1p-41; o.sink_delta["tap"] = 0x1p-42;
  expect(verifyNoiseCertificate(g, o, &p).probabilistic_cert_issued, "exact total budget equality rejected");
  o.sink_delta["tap"] = std::nextafter(0x1p-42, 1.);
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.all_sinks_margin_passed && !r.event_budget_passed && !r.accepts(), "sub-ulp excess lost in budget sum");
  o = certify(g);
  p.site = [](const SiteQuery &q) -> Bound { return bound(q.port_or_node_id == "tap" ? 4096 : 0); };
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.all_contracts_present && r.event_budget_passed && r.sinks[0].margin_passed &&
         !r.sinks[1].margin_passed && !r.probabilistic_cert_issued, "failed second sink was ignored");
  p.site = [](const SiteQuery &q) -> Bound {
    if (q.port_or_node_id == "w" && q.role == Role::Encode) return std::nullopt;
    return bound();
  };
  r = verifyNoiseCertificate(g, o, &p);
  expect(!r.accepts() && r.missing_contract_terms.contains("node/w/encode") && !r.total_delta_valid,
         "omitted required site treated as exact");
  for (auto evidence : {Evidence::Unverified, Evidence::Empirical}) {
    p.site = [=](const SiteQuery &) -> Bound { auto b = bound(); b.evidence = evidence; return b; };
    expect(!verifyNoiseCertificate(g, o, &p).accepts(), "profile or empirical bounds issued certificate");
  }
  p.site = [](const SiteQuery &) -> Bound { auto b = bound(); b.event = "local"; return b; };
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "unrelated local conditional law");
  p = proofs(); o.sink_delta.erase("tap");
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "missing sink allocation");
  o = certify(g); o.target_failure_prob = 0x1p-39;
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "target weaker than theorem cap");
}

void physicalAndRounding() {
  constexpr uint64_t q = 180143985094819841ULL; // prime, 1 mod 32, not exactly binary64
  Topology g{{{"K", 16, {97, q}, {257}}, {"T", 8, {97}, {257}}},
             {source("u", "u", 1ULL << 57), node("rs", Op::Rescale, {0}, 0),
              node("down", Op::Down, {1}, 0)}, {{"out", 2}}};
  g.nodes[2].domain = 1; g.nodes[2].evk = "K-to-T";
  g.nodes[0].source_bound = 1;
  auto p = proofs(); noGroup(p);
  p.site = [](const SiteQuery &s) -> Bound { return bound(0, s.role == Role::Project ? 1 : 0); };
  auto o = certify(g); o.precision_bits = 0;
  auto r = verifyNoiseCertificate(g, o, &p);
  expect(r.errors.empty() && r.sinks.size() == 1, "physical replay error");
  // Independent integer nearest-even oracle: 2^57/q is in [1/2,1),
  // 128 significant bits mean rounding 2^185/q to an integer. No tie.
  mpz_class prime(std::to_string(q));
  mpz_class mant = ((mpz_class(1) << 186) + prime) / (2 * prime);
  Exact physical(mant, mpz_class(1) << 128); physical.canonicalize();
  const auto &s = r.sinks[0];
  expect(s.physical_scale_exact == physical.get_str() && s.ring_n == 8 &&
         r.edge_inventory[1].physical_scale_exact == s.physical_scale_exact,
         "integer divisor, backend rounding or post-down ring/scale wrong");
  expect(physical != roundCKKSScale(Exact(mpz_class(1) << 57) / exactBinary(double(q))), "bad large-prime fixture");
  auto tail = [](int n) { return std::sqrt(2.L) * std::sqrt(2.L * std::log(2.L * n / 0x1p-44L)); };
  expect(*s.certified_bound_units >= tail(8) - 1e-15L && *s.certified_bound_units < tail(16), "tail used source instead of sink ring");
  g.nodes[1].op = Op::Drop;
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.sinks[0].physical_scale_exact == std::to_string(1ULL << 57), "drop divided the scale");

  Topology one{{{"K", 8, {97, 193}, {241}}}, {source("u", "u")}, {{"out", 0}}};
  p.site = [](const SiteQuery &s) -> Bound { return bound(s.role == Role::Output ? 1 : 0); };
  double upper = std::sqrt(2.0);
  auto mid = Exact((exactBinary(upper) + exactBinary(std::nextafter(upper, 0.))) / 2);
  Exact below = mid + Exact(1, mpz_class(1) << 60);
  expect(below * below < 2 && below > mid, "incorrect boundary fixture");
  one.nodes[0].effective_encoder_scale = one.nodes[0].encoded_scale = below;
  r = verifyNoiseCertificate(one, o, &p);
  expect(r.all_contracts_present && !r.sinks[0].margin_passed, "nearest rounding manufactured a passing margin");
  one.nodes[0].effective_encoder_scale = one.nodes[0].encoded_scale = exactBinary(upper);
  expect(verifyNoiseCertificate(one, o, &p).probabilistic_cert_issued, "upper adjacent representable tolerance rejected");

  auto joined = fixture(); joined.nodes[1].encoded_scale = below; joined.nodes[0].encoded_scale = exactBinary(upper);
  for (auto &n : joined.nodes) n.effective_encoder_scale = n.encoded_scale;
  r = verifyNoiseCertificate(joined, o, &p);
  expect(!r.accepts() && r.missing_contract_terms.contains("node/sum/unequal_physical_join_transfer"),
         "rounded-to-equal scales signed a certificate");
}

void dependentAndNonlinearSites() {
  Topology g{{{"K", 8, {97, 193}, {241}}},
             {source("u", "u"), node("r0", Op::Relin, {0}), node("r1", Op::Relin, {0}),
              node("sum", Op::Add, {1, 2})}, {{"out", 3}}};
  for (int i : {1, 2}) { g.nodes[i].k = 1; g.nodes[i].evk = "K/relin"; }
  auto p = proofs(); noGroup(p); auto o = certify(g);
  p.site = [](const SiteQuery &s) -> Bound {
    return bound(0, s.role == Role::KeySwitch ? 4 : s.role == Role::GadgetTrunc ? 9 : 0);
  };
  auto r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].total_variance_proxy_v == 100 &&
         r.sink_forms.at("out").dependentTerms().size() == 4, "shared EVK or nonzero-k treated as fresh");
  auto sw = p.site;
  p.site = [=](const SiteQuery &s) -> Bound {
    if (s.role == Role::GadgetTrunc && s.node.id == "r1") return std::nullopt;
    return sw(s);
  };
  r = verifyNoiseCertificate(g, o, &p);
  expect(!r.accepts() && r.missing_contract_terms.contains("node/r1/gadget_trunc"), "missing k-dependent site");
  g.nodes = {source("u", "u"), node("rot", Op::Rotate, {0}), node("diff", Op::Sub, {1, 0})};
  g.nodes[1].evk = "K/rot1"; g.nodes[1].rotation = 1; g.outputs[0].value = 2;
  p.site = [](const SiteQuery &s) -> Bound {
    return bound(0, s.role == Role::Encrypt || s.role == Role::Permute ? 1 : 0);
  };
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].total_variance_proxy_v == 4,
         "rotation cancelled a different component with its original");
  g.nodes = {source("u", "u"), node("prod", Op::CtMul, {0, 0}), node("rs", Op::Rescale, {1}, 0)};
  p.site = [](const SiteQuery &s) -> Bound {
    if (s.role == Role::Product) return std::nullopt;
    return bound(0, s.role == Role::RescaleRound ? 1 : 0);
  };
  r = verifyNoiseCertificate(g, o, &p);
  expect(!r.accepts() && r.missing_contract_terms.contains("node/prod/ct_mul_including_cross_term"),
         "nonlinear cross term assumed sub-Gaussian");
  // A signed plaintext multiplier that is exactly -1 cancels the shared
  // input on re-addition; it still owes an encoding and multiplication proof.
  g.nodes = {source("u", "u"), node("minus", Op::ScalarMulPlain, {0}), node("sum", Op::Add, {1, 0})};
  g.nodes[1].real_scalar = -1; g.nodes[1].effective_encoder_scale = g.nodes[1].encoded_scale = 1;
  p.site = [](const SiteQuery &s) -> Bound { return bound(0, s.role == Role::Encrypt ? 5 : 0); };
  r = verifyNoiseCertificate(g, o, &p);
  expect(r.probabilistic_cert_issued && r.sinks[0].total_variance_proxy_v == 0, "lost signed scalar gain");
  g.nodes[1].real_scalar = 0;
  p.site = [](const SiteQuery &s) -> Bound { return s.role == Role::PlainEncode ? Bound{} : Bound(bound()); };
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "zero multiplier erased a required primitive");
}

void productionScaleBranches() {
  const auto meta = plannedEncodingScale(0.5);
  expect(meta == Exact("240615969168004511545033772477625056927/170141183460469231731687303715884105728") &&
         effectiveEncoderScale(meta, 53) == Exact("6369051672525773/4503599627370496") &&
         effectiveEncoderScale(meta, 60) == meta, "encoder branch versus 128-bit metadata");
  auto g = fixture();
  g.nodes[0] = source("left", "L", 4); g.nodes[0].source_bound = 3;
  g.nodes[1] = source("right", "R", 9); g.nodes[1].source_bound = 5;
  auto p = proofs(); noGroup(p);
  auto r = verifyNoiseCertificate(g, certify(g), &p);
  auto &join = r.edge_inventory.at(2);
  expect(r.errors.empty() && join.join_branch == "multiply_left" && join.multiplier_a == "2" &&
         join.multiplier_b == "1" && join.residual_a == "-1" && join.scale_bias == "3" &&
         r.sink_forms.at("out").linearTerms().at("L").signed_coeff == 2,
         "actual AddNew integer multiplier or ideal-message charge lost");
  g.nodes[2].inputs = {1, 0}; r = verifyNoiseCertificate(g, certify(g), &p);
  expect(r.edge_inventory.at(2).join_branch == "multiply_right" && r.edge_inventory.at(2).scale_bias == "3",
         "asymmetric AddNew alignment");
  g.nodes[2].op = Op::Sub; r = verifyNoiseCertificate(g, certify(g), &p);
  expect(r.edge_inventory.at(2).residual_b == "-1" && r.edge_inventory.at(2).scale_bias == "3" &&
         r.sink_forms.at("out").linearTerms().at("L").signed_coeff == -2,
         "v1 stores pre-subtraction residuals and signs the propagated gain once");
  g.nodes[2].op = Op::Add;
  g.nodes[1].effective_encoder_scale.reset(); r = verifyNoiseCertificate(g, certify(g), &p);
  expect(!r.accepts() && r.missing_contract_terms.contains("node/right/missing_encoding_scale_binding") &&
         r.sinks[0].physical_scale_exact.empty(), "missing effective source binding defaulted to metadata");

  Topology hi{{{"K", 8, {97, 113, 193}, {241}, 2}},
              {source("u", "u"), node("rs", Op::Rescale, {0}, 0)}, {{"out", 1}}};
  hi.nodes[0].level = 2; hi.nodes[0].source_bound = 1;
  r = verifyNoiseCertificate(hi);
  auto expected = roundCKKSScale(roundCKKSScale(Exact(65536, 193)) / 113);
  expect(r.errors.empty() && r.sinks.at(0).physical_scale_exact == expected.get_str(), "two-prime sequential rescale");
  hi.nodes[1].level = 1;
  expect(!verifyNoiseCertificate(hi).errors.empty(), "wrong rescale count accepted");
}

void malformedInputs() {
  auto p = proofs(); auto o = certify(fixture());
  for (auto edit : std::vector<std::function<void(Topology &)>>{
      [](auto &g) { g.outputs.clear(); }, [](auto &g) { g.outputs.push_back(g.outputs[0]); },
      [](auto &g) { g.outputs[0].value = 0; }, [](auto &g) { g.nodes[1].id = g.nodes[0].id; },
      [](auto &g) { g.nodes[2].inputs = {0, 2}; }, [](auto &g) { g.nodes[2].k = 1; },
      [](auto &g) { g.nodes[2].level = 0; }, [](auto &g) { g.domains[0].ring_n = 6; },
      [](auto &g) { g.domains[0].p = {97}; }, [](auto &g) { g.domains[0].q[1] = 289; },
      [](auto &g) { g.nodes[0].encoded_scale = 0; }, [](auto &g) { g.nodes[0].encoded_scale = Exact(1, 0); },
      [](auto &g) { g.nodes[2].op = static_cast<Op>(1234); }}) {
    auto g = fixture(); edit(g);
    auto r = verifyNoiseCertificate(g, o, &p);
    expect(!r.accepts() && !r.errors.empty(), "malformed concrete topology accepted");
  }
  auto g = fixture(); g.nodes[0].encoded_scale.reset();
  auto missing = verifyNoiseCertificate(g, o, &p);
  expect(!missing.accepts() && !missing.sinks[0].actual_physical_scale, "substituted nominal scale");
  for (double bad : {-1., std::numeric_limits<double>::infinity(), unset}) {
    g = fixture(); p = proofs();
    p.site = [=](const SiteQuery &) -> Bound { return bound(0, bad); };
    expect(!verifyNoiseCertificate(g, o, &p).accepts(), "invalid variance accepted");
    p.site = [=](const SiteQuery &) -> Bound { return bound(bad, 0); };
    expect(!verifyNoiseCertificate(g, o, &p).accepts(), "invalid bias accepted");
    p.site = [=](const SiteQuery &) -> Bound { auto b = bound(); b.failures["wrap"] = bad; return b; };
    expect(!verifyNoiseCertificate(g, o, &p).accepts(), "invalid event accepted");
    auto other = o; other.sink_delta["out"] = bad; p = proofs();
    expect(!verifyNoiseCertificate(g, other, &p).accepts(), "invalid tail allocation accepted");
  }
  auto other = o; other.sink_delta["out"] = 0;
  expect(!verifyNoiseCertificate(g, other, &p).accepts(), "zero tail allocation");
  p.common = [](const Topology &) -> std::optional<CommonLaw> {
    return CommonLaw{Evidence::Empirical, "G", "1000 trials", 0, {}, {}};
  };
  expect(!verifyNoiseCertificate(g, o, &p).accepts(), "empirical common law accepted");
}
} // namespace

int main() {
  try {
    affineArithmetic(); jointAndFanout(); budgetsAndEvidence();
    physicalAndRounding(); dependentAndNonlinearSites(); malformedInputs();
    productionScaleBranches();
    std::cout << "Pass2AffineNoiseTest: passed\n";
  } catch (const std::exception &e) {
    std::cerr << "Pass2AffineNoiseTest: " << e.what() << '\n';
    return 1;
  }
}

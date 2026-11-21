#include "ir/NoiseCertVerifier.h"
#include <iostream>

using namespace snabf;
using namespace snabf::ir::noise;
namespace {
void expect(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
Node op(std::string id, Op kind, std::vector<size_t> inputs, int level = 1) {
  Node n{};
  n.id = id; n.op = kind; n.inputs = inputs; n.level = level;
  return n;
}
const ScaleDomain domain{"k", 32, 39, {193, 257}, {449}};
BoundPlainCall encode(const Node &n, const Exact &requested, std::span<const double> payload) {
  bool source = n.op == Op::Source;
  return BoundPlainCall::encode({"encode:" + n.id, n.id, source ? 0 : 1, n.level, 16,
                                domain, true, "[]float64", 53, 53, source}, requested, payload);
}
Topology start() {
  auto u = op("u", Op::Source, {});
  u.input_draw = "input0"; u.source_bound = 1;
  const auto call = encode(u, Exact(1) + Exact(0x1p-53), {}).receipt();
  u.encoded_scale = call.realized_meta_scale; u.effective_encoder_scale = call.realized_eff_scale;
  return {{{domain.key_id, domain.ring_n, domain.q, domain.p}}, {u}, {}};
}
Options options(const Topology &g) {
  Options o; o.b2i = true; o.mode = Mode::Probability;
  for (const auto &s : g.outputs) o.sink_delta.emplace(s.id, 0x1p-45);
  return o;
}
const LocalTransfer &row(const CertAuditReport &r, const std::string &site) {
  for (const auto &t : r.local_transfers) if (t.site == "node/" + site) return t;
  throw std::runtime_error("no local transfer: " + site);
}
void linear() {
  auto g = start();
  g.nodes.push_back(op("twice", Op::Add, {0, 0}));
  g.nodes.push_back(op("minus", Op::Neg, {0}));
  g.nodes.push_back(op("zero", Op::Sub, {1, 1}));
  g.nodes.push_back(op("same", Op::Drop, {2}));
  g.nodes.push_back(op("drop", Op::Drop, {2}, 0));
  g.nodes.push_back(op("dropZero", Op::Drop, {3}, 0));
  g.nodes.push_back(op("plusZero", Op::Add, {3, 4}));
  g.outputs = {{"d", 5}, {"z", 6}, {"p", 7}};
  auto o = options(g);
  auto r = verifyNoiseCertificate(g, o);
  expect(r.errors.empty() && !r.accepts() && !r.total_delta_valid && r.common_event.empty(), "spurious law");
  auto &add = row(r, "twice/add"); auto &drop = row(r, "drop/level_reduce");
  expect(!add.blocker.empty() && add.candidate_coeff_abs == 193*257-1 && !add.bias, "unproved add");
  expect(!drop.blocker.empty() && drop.active_q == 193 && !drop.bias, "wrong output Q or assumed ideal coeffs");
  expect(row(r, "minus/neg").lhs_gain == -1 && row(r, "minus/neg").bias == 0, "negate centered lift");
  expect(row(r, "zero/sub").candidate_coeff_abs == 0 && row(r, "zero/sub").rhs_gain == -1 &&
         row(r, "dropZero/level_reduce").bias == 0 && row(r, "same/level_reduce").bias == 0 &&
         row(r, "plusZero/add").bias == 0, "lost proven centered identity");
  expect(r.sink_forms.at("z").missingContracts().contains("node/twice/add") &&
         r.sink_forms.at("z").missingContracts().contains("node/u/encrypt"), "cancellation erased history");
  AnalyticContracts legacy;
  expect(!verifyNoiseCertificate(g, o, &legacy).errors.empty(), "mixed incompatible provider and gains");
}
void fixedPoint(bool add) {
  auto g = start();
  auto n = op("pt", add ? Op::AddPlain : Op::ScalarMulPlain, {0});
  const double w = -1.75;
  if (add) n.plain_values = {w}; else n.real_scalar = w;
  n.bound_plain_call = encode(n, 1, {&w, 1});
  n.encoded_scale = n.bound_plain_call->receipt().realized_meta_scale;
  n.effective_encoder_scale = n.bound_plain_call->receipt().realized_eff_scale;
  g.nodes.push_back(n); g.outputs = {{"s", 1}};
  auto o = options(g); auto r = verifyNoiseCertificate(g, o);
  auto &enc = row(r, "pt/plain_encoding_at_output");
  auto a = *g.nodes[0].encoded_scale;
  expect(enc.encoded_constant == -2 && enc.delta_pt == Exact(-1,4) && enc.lhs_gain == (add ? 1 : -2), "wrong signed DC");
  // No mock common law. The partial affine form contains exactly the known
  // deterministic terms, including the previously charged metadata drift.
  Exact sourceDrift = a-1, localDrift = add ? Exact(abs(1-a)*abs(Exact(w))) : Exact(0);
  Exact bias = add ? Exact(1,4) : Exact(a/4);
  expect(enc.bias == bias && r.sink_forms.at("s").bias() == (add ? 1 : 2)*sourceDrift + localDrift + bias,
         "dropped delta*e or double-counted the B1 drift");
  expect(!r.sinks[0].deterministic_bias_b && !r.sinks[0].certified_bound_units && !r.accepts(), "partial bias issued");
  if (add) g.nodes[1].plain_values[0] = -1.5; else g.nodes[1].real_scalar = -1.5;
  expect(!verifyNoiseCertificate(g, o).errors.empty(), "reused receipt on other public bits");
  g.nodes[1] = n; g.nodes[1].id = "pt-other";
  expect(!verifyNoiseCertificate(g, o).errors.empty(), "reused receipt on other call");
}
} // namespace
int main() {
  linear(); fixedPoint(false); fixedPoint(true);
  std::cout << "B2-i centered lifts, signed gain, bias, binding and refusals passed\n";
}

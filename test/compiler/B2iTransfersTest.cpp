#include "ir/NoiseCertVerifier.h"
#include "ir/CommonArithmeticSupport.h"
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
  return {{{domain.key_id, domain.ring_n, domain.q, domain.p}}, {u}, {}, 16};
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

Node tile(std::string id, Op kind, size_t parent, int level, std::vector<double> v) {
  Node n = op(std::move(id), kind, {parent}, level);
  n.plain_values = std::move(v);
  n.bound_plain_call = encode(n, 1, n.plain_values);
  const auto &call = n.bound_plain_call->receipt();
  n.encoded_scale = call.realized_meta_scale; n.effective_encoder_scale = call.realized_eff_scale;
  return n;
}
void uniformWords() {
  auto g = start();
  g.nodes.push_back(tile("mask", Op::VectorMulPlain, 0, 1, {200,200,200,200}));
  g.outputs = {{"s",1}};
  auto run = [&] { return verifyNoiseCertificate(g, options(g)); };
  auto r = run();
  const auto &enc = row(r, "mask/plain_encoding_at_output");
  expect(r.errors.empty() && enc.encoded_constant == 200 && enc.delta_pt == 0 &&
         enc.ntt_word_upper == 200+2*193*5 && enc.formula == "b2i/dc-word-bounded-ntt-v6.2.0", "unreduced positive limb");
  expect(!row(r, "mask/mul_plain").blocker.empty() && !r.accepts() && !r.sinks[0].certified_bound_units,
         "used plaintext proof as a ciphertext wrap guarantee");
  expect(r.sink_forms.at("s").bias() == 200*(*g.nodes[0].encoded_scale-1), "vector lost parent bias");
  g.direct_go_slots = 8;
  expect(!run().errors.empty(), "accepted a different layout for the NTT stage count");
  g.direct_go_slots = 16;
  g.nodes[1] = tile("mask", Op::VectorMulPlain, 0, 1, {-1.75,-1.75});
  r = run();
  const Exact a = *g.nodes[0].encoded_scale;
  expect(row(r, "mask/plain_encoding_at_output").lhs_gain == -2 &&
         r.sink_forms.at("s").bias() == 2*(a-1)+a/4, "lost signed vector gain or double-counted delta*e");
  g.nodes[1] = tile("mask", Op::VectorMulPlain, 0, 1, {0.,0.,0.,0.});
  g.nodes[1].plain_values[2] = -0.;
  expect(!run().errors.empty(), "accepted stale interior signed-zero word");
  g.nodes[1] = tile("mask", Op::VectorMulPlain, 0, 1, {0.,0.,-0.,0.});
  r = run();
  expect(row(r, "mask/plain_encoding_at_output").blocker == "vector_ifft_not_proved" &&
         !row(r, "mask/mul_plain").lhs_gain, "silently collapsed a nonidentical bit pattern");
  g.nodes[1] = tile("mask", Op::VectorMulPlain, 0, 1, {-386.,-386.});
  r = run(); // negative exact multiple writes q, not 0, on the 193 limb
  expect(row(r, "mask/plain_encoding_at_output").ntt_word_upper == 193+2*193*5 &&
         row(r, "mask/plain_encoding_at_output").encoded_constant == -386, "negative multiple fast CRT");
}

void nonuniform() {
  auto g = start();
  auto m = tile("mask", Op::VectorMulPlain, 0, 1, {97,-97,97,-97});
  g.nodes.push_back(m); g.outputs = {{"s",1}};
  auto run = [&] { return verifyNoiseCertificate(g, options(g)); };
  expect(row(run(), "mask/plain_encoding_at_output").blocker == "vector_ifft_not_proved", "borrowed mask provenance");
  g.nodes[1].diagonal_mask = true;
  auto r = run(); const auto &enc = row(r, "mask/plain_encoding_at_output");
  expect(enc.diagonal && enc.diagonal->period == 2 && enc.diagonal->stride == 8 && enc.diagonal->coeff_abs == 97 &&
         enc.bias && *enc.bias > 0 && !enc.delta_pt && !enc.lhs_gain && !enc.encoded_constant &&
         row(r, "mask/mul_plain").blocker == "need_tighter_centered_decryption_coefficients" && !r.accepts(), "false signed/whole-op proof");
  auto &edge = g.nodes[1]; edge.bound_plain_call.reset();
  expect(row(run(), "mask/plain_encoding_at_output").blocker == "missing_bound_encoding_call", "unbound mask accepted");
  edge = m; edge.diagonal_mask = true; edge.effective_encoder_scale = 2;
  expect(!run().errors.empty(), "ignored E");
  edge = m; edge.diagonal_mask = true; edge.plain_values[1] = -96;
  expect(!run().errors.empty(), "stale public coefficient");
  edge = tile("mask", Op::VectorMulPlain, 0, 0, {194,196}); edge.diagonal_mask = true; // exact DC 195 -> 2
  g.nodes[0].level = 0;
  r = run();
  expect(row(r,"mask/plain_encoding_at_output").blocker == "plaintext_centered_lift_unproved" &&
         !row(r,"mask/plain_encoding_at_output").diagonal, "borrowed an inactive prime for plaintext centering");
}

void exactProduct(int sign) {
  auto g = start();
  g.nodes.push_back(op("zero", Op::Sub, {0,0}));
  g.nodes.push_back(op("drop", Op::Drop, {1}, 0));
  const double v = sign*96.;
  g.nodes.push_back(tile("base", Op::AddPlain, 2, 0, {v,v}));
  g.nodes.push_back(tile("wrap", Op::VectorMulPlain, 3, 0, {5,5,5,5}));
  g.nodes.push_back(tile("next", Op::VectorMulPlain, 4, 0, {-1,-1}));
  g.outputs = {{"s",5}};
  auto r = verifyNoiseCertificate(g, options(g));
  const auto &wrap = row(r, "wrap/mul_plain");
  expect(r.errors.empty() && wrap.formula == "b2i/exact-dc-recenter-v1" &&
         wrap.dc_before_center == sign*480 && wrap.dc_after_center == sign*94 &&
         wrap.dc_center_correction == -sign*386 && wrap.bias == 386, "wrong signed two-modulus correction");
  const auto &next = row(r, "next/mul_plain");
  expect(next.dc_before_center == -sign*94 && next.bias == 0, "propagated the unreduced coefficient");
  const auto &f = r.sink_forms.at("s");
  const Exact drift = *g.nodes[0].encoded_scale-1; // source cancels; AddPlain metadata mismatch remains
  expect(f.bias() == 386 + 480*drift && f.missingContracts().contains("node/u/encrypt") &&
         !r.total_delta_valid && !r.sinks[0].certified_bound_units && !r.accepts(), "lost bias/history or false issuance");
  // Same public multiplier and modulus but a secret parent, not a known DC.
  g.nodes[1] = op("zero", Op::Neg, {0});
  r = verifyNoiseCertificate(g, options(g));
  expect(!row(r, "wrap/mul_plain").dc_center_correction && !row(r, "wrap/mul_plain").bias &&
         !row(r, "wrap/mul_plain").blocker.empty(), "inferred signed wrap from a mere bound");
}

void unsafeWordRange() {
  std::set<int64_t> used;
  const int n = 1024;
  const auto q = uint64_t(pass3::nearestPrime(n, 60, used));
  const ScaleDomain d{"wide", n, 39, {q}, {uint64_t(pass3::nearestPrime(n, 45, used))}};
  auto g = start(); g.domains = {{d.key_id, n, d.q, d.p}}; g.nodes[0].level = 0; g.direct_go_slots = n/2;
  auto m = op("tiny", Op::VectorMulPlain, {0}, 0); m.plain_values = {-.25,-.25};
  m.bound_plain_call = BoundPlainCall::encode({"encode:tiny", "tiny", 1, 0, n/2, d, true,
                                               "[]float64", 53, 53, false}, 1, m.plain_values);
  m.effective_encoder_scale = 1; m.encoded_scale = 1;
  g.nodes.push_back(m); g.outputs = {{"s",1}};
  auto r = verifyNoiseCertificate(g, options(g));
  const auto &enc = row(r, "tiny/plain_encoding_at_output");
  expect(r.errors.empty() && enc.blocker == "unproved_ntt_word_range" && !enc.encoded_constant && !enc.ntt_word_upper,
         "overflowed or silently normalized the raw q word");
}

void freshDegreeOne() {
  const ScaleDomain d{"retained",16384,39,{1146881,540311683073,540311912449},{56863324045313}};
  const Exact E = Exact(0x1p39);
  Topology g; g.domains={{d.key_id,d.ring_n,d.q,d.p,1}}; g.direct_go_slots=4096; g.fresh_width=64;
  auto u=op("u",Op::Source,{},2); u.input_draw=u.id; u.source_bound=1;
  auto bind=[&](Node &n,bool dynamic,const std::string &id) {
    n.bound_plain_call=BoundPlainCall::encode({"encode:"+id,id,dynamic?0:1,n.level,g.direct_go_slots,d,true,
        "[]float64",53,53,dynamic},E,n.plain_values);
    n.encoded_scale=E; n.effective_encoder_scale=E;
  };
  bind(u,true,u.id);
  auto rot=op("r",Op::Rotate,{0},2); rot.rotation=1; rot.evk="retained:rotation:1";
  auto v=op("v",Op::VectorMulPlain,{1},2); v.diagonal_mask=true; v.plain_values={.25,-.25};bind(v,false,v.id);
  g.nodes={u,rot,v,op("rs",Op::Rescale,{2},1)}; g.outputs={{"sink",3}};
  auto audit=[&] {return verifyNoiseCertificate(g,options(g));};
  auto r=audit();
  expect(r.errors.empty() && !r.accepts() && row(r,"v/mul_plain").bias>0 && !row(r,"v/mul_plain").lhs_gain &&
         row(r,"rs/rescale_round").lhs_gain==Exact(1, d.q.back()) && !r.total_delta_valid &&
         r.missing_contract_terms.contains("node/r/evk_switch"),"wrong fresh/correlated support");
  auto noLift=[&] {auto t=audit(); expect(t.errors.empty() && row(t,"v/mul_plain").formula.empty() &&
       row(t,"rs/rescale_round").formula.empty() && !t.accepts(),"unbound support released");};
  g.nodes[0].bound_plain_call.reset(); noLift();
  bind(g.nodes[0],true,"wrong"); noLift();
  g.nodes[0]=u;g.nodes[0].input_draw="another";noLift();g.nodes[0]=u;
  g.nodes[0].source_bound=2; noLift();g.nodes[0]=u;
  g.nodes[1].evk="another:rotation:1";noLift();g.nodes[1]=rot;
  // Inject a hypothesis into the private transfer ONLY, not the checker:
  // both sufficient guards accept the largest possible bound and reject +1.
  std::vector<LocalStep> done(3);
  auto boundary=[&](size_t idx, const LocalTransfer &proof) {
    const auto p=g.nodes[idx].inputs[0];const Exact limit(proof.premises.at("max_parent_coeff"));
    for (int j : {0,1}) {
      done[p].fresh_coeff=limit+j;
      auto step=boundedStep(g,idx,done,E,Exact(1),true,1,1);
      const auto &end=step.rows.back();
      expect(bool(step.fresh_coeff)==(j==0) && bool(end.bias)==(j==0) && end.formula.empty()==(j!=0),
             "off by one in strict centered modulus guard");
    }
  };
  boundary(2,row(r,"v/mul_plain")); boundary(3,row(r,"rs/rescale_round"));
}
// Isolate the transfer lemma; no production entry point accepts these hypotheses.
void commonBoundaries() {
  const Exact q=1146881, half=(q-1)/2;
  Topology g; g.fresh_width=4;
  g.domains={{"k",16384,{1146881,540311683073},{56863324045313},1}};
  g.nodes={op("x",Op::Source,{},0),op("y",Op::Source,{},0),op("z",Op::Add,{0,1},0)};
  std::vector<LocalStep> steps(3);
  auto &x=steps[0].arithmetic;auto &y=steps[1].arithmetic;auto &z=g.nodes[2];
  auto test=[&](const Exact &ideal) {
    steps[2]=LocalStep{};
    commonArithmeticSupport(g,2,steps,Exact(1),ideal,Exact(0),1,1);
    return steps[2].arithmetic;
  };
  y=ArithmeticSupport{1,1,1,1};
  x=ArithmeticSupport{1,half-2,half,1};
  expect(test(q)->coeff==half-1,"largest unambiguous integral lift refused");
  x->coeff=half-1;
  expect(!test(q),"accepted CRT display endpoint +floor(Q/2)");
  x->coeff=half;
  expect(!test(q) && steps[2].rows.at(0).blocker=="unproved_common_centered_lift" &&
         !steps[2].rows[0].bias,"accepted half-Q plus one via a loose embedding");
  // Independent second route, using all odd roots before centering. Integer
  // coefficients are essential: a nonintegral upper just below half is OK.
  expect(test(half-3+Exact(3,4))->coeff==half-1 && !test(half-2),"inverse embedding off by one");
  z.op=Op::Sub;z.inputs={0,0};
  expect(test(q)->error==0 && steps[2].arithmetic->coeff==0,"identical SSA did not cancel");
  z.inputs={0,1};x=ArithmeticSupport{1,1,0,1};y=x;
  expect(test(0)->error==2,"cancelled distinct SSA by common key");
  z.op=Op::CtMul;z.inputs={0,0};
  auto sq=test(0);expect(sq && sq->degree==2 && sq->coeff==1 && sq->error==1 &&
      steps[2].rows[0].centered_support.at("cross")=="1","lost same-draw nonlinear cross term");
  x->degree=2;expect(!test(q),"accepted degree-two ciphertext multiply");
  x.reset();expect(!test(q),"accepted a missing parent");
  x=sq;z=op("z",Op::Relin,{0},1);g.nodes[0].level=1;z.k=1;z.evk="k:relin";
  auto rel=test(0);expect(rel && rel->degree==1 && steps[2].rows.size()==2,"lost relin obligations");
  const Exact N=g.domains[0].ring_n;
  expect(steps[2].rows[0].centered_support.at("cut_coeff")==Exact(N*N*half).get_str(),"used s not s squared");
  z.evk="other:relin";expect(!test(0),"accepted unrelated relinearization key");z.evk="k:relin";
  z.k=0;expect(!test(0),"unmodelled native degree-two switch");z.k=1;
  x->degree=1;expect(!test(0),"relinearized degree one");
  z=op("z",Op::Rotate,{0},1);z.rotation=-1;z.evk="k:rotation:-1";
  expect(test(0).has_value() && steps[2].rows.size()==2,"signed rotation K0");
  z.k=1;expect(test(0) && steps[2].rows.size()==3,"rotation K1");
  z.evk="k:rotation:1";expect(!test(0),"rotation with wrong signed EVK");
  z.evk="k:rotation:-1";z.k=2;expect(!test(0),"unknown projection");
  z.k=1;x->degree=2;expect(!test(0),"rotation of degree two");
  z=op("z",Op::Rescale,{0},0);expect(!test(0),"rescaled degree two");
  x->degree=1;g.nodes[0].level=0;expect(!test(0),"rescale without actual dropped suffix");
  g.nodes[0].level=1;g.domains[0].rescale_primes=2;expect(!test(0),"unexpected double-prime rescale");
}
} // namespace
int main() {
  linear(); fixedPoint(false); fixedPoint(true);
  uniformWords(); nonuniform(); exactProduct(1); exactProduct(-1); unsafeWordRange();
  freshDegreeOne();commonBoundaries();
  std::cout << "B2-i public DC encodings, centered transfers, signed bias and refusals passed\n";
}

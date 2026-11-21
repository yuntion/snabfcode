// Inspection-only differential test adapter. Never seals or lowers IR. The
// original source is hashed for identity; Python checks its actual semantics.
#include "ir/NativeContracts.h"
#include "ir/NoiseCertIR.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace snabf;
using namespace snabf::ir;
namespace j = llvm::json;
namespace {
using T = ExactScaleTracker;
using R = ExactRational;
j::Array primes(const std::vector<uint64_t> &x) {
  j::Array a; for (auto p : x) a.push_back(int64_t(p)); return a;
}
j::Value text(const std::optional<R> &v) { return v ? j::Value(v->get_str()) : j::Value(nullptr); }
j::Value upper(const std::optional<double> &v) {
  return v ? j::Value(llvm::utohexstr(std::bit_cast<uint64_t>(*v), true, 16)) : j::Value(nullptr);
}
j::Object callRow(const V1EncodingCall &v) {
  const auto &c = v.receipt; const auto &d = c.domain;
  return j::Object{{"call_id",c.call_id},{"site_id",c.site_id},{"operand_index",c.operand_index},
      {"level",c.level},{"slots",c.slots},{"key_id",d.key_id},{"ring_n",d.ring_n},
      {"default_log_scale",d.default_log_scale},{"q",primes(d.q)},{"p",primes(d.p)},
      {"dynamic_input",c.dynamic_input},{"is_batched",c.is_batched},{"arg_type",c.arg_type},
      {"arg_precision_bits",c.arg_precision_bits},{"encoder_prec_bits",c.encoder_prec_bits},
      {"payload_count",int64_t(c.payload_count)},{"payload_sha256",c.payload_sha256},
      {"argument_sha256",c.argument_sha256 ? j::Value(*c.argument_sha256) : j::Value(nullptr)},
      {"branch",c.branch},{"quantization_rule",c.quantization_rule},{"requested_scale",c.requested_scale.get_str()},
      {"log_argument_exact",v.log_argument.get_str()},{"realized_meta_scale",c.realized_meta_scale.get_str()},
      {"realized_eff_scale",c.realized_eff_scale.get_str()}};
}
j::Object rescaleRow(const RescaleStepReceipt &r) {
  j::Array div;
  for (const auto &x : r.divisions)
    div.emplace_back(j::Object{{"q",int64_t(x.q)},{"metadata",x.after.metadata.get_str()},
                               {"effective",x.after.effective.get_str()}});
  return j::Object{{"in_level",r.in_level},{"out_level",r.out_level},{"levels_consumed",r.levels_consumed},
      {"mul_depth_delta",r.mul_depth_delta},{"dropped_q",primes(r.dropped_q)},{"meta_before",r.meta_before.get_str()},
      {"meta_after",r.meta_after.get_str()},{"eff_before",r.eff_before.get_str()},{"eff_after",r.eff_after.get_str()},
      {"divisions",std::move(div)}};
}
j::Object joinRow(const AddSubScaleAnalysis &r) {
  return j::Object{{"out_meta_scale",r.out_meta_scale.get_str()},{"lhs_int_mul",r.lhs_int_mul},{"rhs_int_mul",r.rhs_int_mul},
      {"common_effective_scale",text(r.common_effective_scale)},{"lhs_signed_effect",r.lhs_signed_effect.get_str()},
      {"rhs_signed_effect",r.rhs_signed_effect.get_str()},{"lhs_bound",text(r.lhs_bound)},{"rhs_bound",text(r.rhs_bound)},
      {"abs_lhs_residual",r.abs_lhs_residual.get_str()},{"abs_rhs_residual",r.abs_rhs_residual.get_str()},
      {"missing_operand_bounds",r.missing_operand_bounds},{"encoded_bias",text(r.encoded_bias)},
      {"decoded_bias",text(r.decoded_bias)},{"encoded_bias_upper",upper(r.encoded_bias_upper_mpfr)},
      {"decoded_bias_upper",upper(r.decoded_bias_upper_mpfr)}};
}
j::Object trace(ModuleOp m, const std::string &sourceSha) {
  require(succeeded(rejectNativeExecution(m)), "native execution unavailable");
  auto plan = field<DictionaryAttr>(m,"snabf.pass3");
  std::map<std::string,std::string> names;
  for (auto a : field<ArrayAttr>(plan,"edges")) {
    auto e=cast<DictionaryAttr>(a);
    names.emplace(field<StringAttr>(e,"ID").str(),field<StringAttr>(e,"op").str());
  }
  std::vector<V1EncodingCall> calls;
  auto g = checkedNoiseTopology(m,calls);
  require(g.domains.size()==1 && !calls.empty(),"expected bound single-ring release program");
  const auto &d = calls.front().receipt.domain;
  auto audit = noise::verifyNoiseCertificate(g); // legacy metadata-reference audit, not a provider
  require(audit.errors.empty() && audit.edge_inventory.size()==g.nodes.size(),"invalid engineering topology");
  struct State { ScaleState s; R bound; bool rebased; };
  std::vector<State> states;
  j::Array edges, callRows;
  for (const auto &c : calls) callRows.emplace_back(callRow(c));
  for (size_t i=0; i<g.nodes.size(); ++i) {
    const auto &n = g.nodes[i]; const auto &old = audit.edge_inventory[i];
    require(n.id==old.id && !old.ideal_bound.empty(),"missing checked magnitude");
    auto op = names.at(n.id.substr(0,n.id.find("#add")));
    j::Array args; bool rebased=false;
    for (size_t a : n.inputs) { args.push_back(g.nodes[a].id); rebased |= states.at(a).rebased; }
    j::Object row{{"node_id",n.id},{"op",op},{"inputs",std::move(args)},{"level",n.level},{"abs_bound",old.ideal_bound}};
    using O = noise::Op;
    auto encoded = [&] { return ScaleState{n.encoded_scale.value(),n.effective_encoder_scale.value()}; };
    auto s = n.op==O::Source ? encoded() : states.at(n.inputs.at(0)).s;
    switch (n.op) {
    case O::Source: case O::Drop: case O::Relin: case O::Rotate: break;
    case O::ScalarMulPlain: case O::VectorMulPlain: s=T::multiply(s,encoded()); break;
    case O::CtMul: s=T::multiply(s,states.at(n.inputs.at(1)).s); break;
    case O::Rescale: {
      int level=g.nodes[n.inputs[0]].level;
      std::vector<uint64_t> tail;
      for (int l=level; l>n.level; --l) tail.push_back(d.q.at(l));
      auto r=T::replayRescale(s,d,level,tail);
      row["rescale"]=rescaleRow(r); s={r.meta_after,r.eff_after}; break;
    }
    case O::Add: case O::AddPlain: case O::Sub: {
      auto &lhs=states.at(n.inputs.at(0)); R bb=0;
      bool plain=n.op==O::AddPlain;
      if (plain) for (double x:n.plain_values) bb=std::max(bb,R(std::abs(x)));
      else bb=states.at(n.inputs.at(1)).bound;
      auto r=T::analyzeAddSubMismatch(n.op==O::Sub ? JoinOp::Sub : JoinOp::Add, s,
          plain ? encoded() : states.at(n.inputs.at(1)).s,lhs.bound,bb);
      row["join"]=joinRow(r);
      rebased |= !r.common_effective_scale;
      s={r.out_meta_scale,r.common_effective_scale.value_or(r.out_meta_scale)}; break;
    }
    default: throw ContractError("unsupported v1 trace opcode: "+op);
    }
    require(s.metadata.get_str()==old.physical_scale_exact,"legacy metadata diverged at "+n.id);
    row["metadata"]=s.metadata.get_str(); row["reference_gain"]=s.effective.get_str(); row["reference_only"]=rebased;
    row["effective"]=rebased ? j::Value(nullptr) : j::Value(s.effective.get_str());
    states.push_back({s,R(old.ideal_bound),rebased}); edges.emplace_back(std::move(row));
  }
  return j::Object{{"schema","snabf.exact-scale.v1"},{"source_sha256",sourceSha},
      {"emit_sha256",field<StringAttr>(m,"snabf.emit_sha256").str()},{"key_id",d.key_id},{"ring_n",d.ring_n},
      {"ordered_q",primes(d.q)},{"ordered_p",primes(d.p)},{"calls",std::move(callRows)},
      {"edges",std::move(edges)},{"probabilistic_cert_issued",false}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc==3,"usage: ExactScaleTraceTest CHECKED.mlir ORIGINAL_SOURCE.mlir");
    DialectRegistry registry; registerAllDialects(registry); registry.insert<snabf::ir::SNABF>();
    MLIRContext ctx(registry); ctx.allowUnregisteredDialects();
    auto mod=parseSourceFile<ModuleOp>(argv[1],&ctx);
    require(bool(mod),"failed to parse checked module");
    auto bytes=llvm::MemoryBuffer::getFile(argv[2]); require(bool(bytes),"missing original source");
    llvm::SHA256 sha; sha.update((*bytes)->getBuffer());
    llvm::outs() << j::Value(trace(*mod,llvm::toHex(sha.final(),true))) << '\n';
  } catch (const std::exception &e) { llvm::errs()<<e.what()<<'\n'; return 1; }
}

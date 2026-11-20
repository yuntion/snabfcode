#include "EmitGraph.h"
#include "pass1/SubgroupTower.h"
#include "pass2/AffineErrorForm.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringSwitch.h"
#include <bit>
#include <cmath>

namespace snabf::ir::lattigo_target::detail {
namespace {
// Canonical hash: uint64 big endian; byte strings and integer vectors have a
// uint64 length. Signed payloads are raw IEEE-754 binary64, not printed floats.
struct Hash {
  llvm::SHA256 h;
  void word(uint64_t x) {
    uint8_t bytes[8];
    for (int i = 7; i >= 0; --i, x >>= 8) bytes[i] = uint8_t(x);
    h.update(ArrayRef<uint8_t>(bytes, 8));
  }
  void text(StringRef s) { word(s.size()); h.update(s); }
  void vector(ArrayRef<int64_t> xs) { word(xs.size()); for (auto x : xs) word(x); }
  std::string finish() { return llvm::toHex(h.final(), true); }
};

DictionaryAttr buildEncodingPlan(EmitGraph &g) {
  Builder b(g.program.getContext());
  Hash program;
  program.text("snabf.encoding-input.v1");
  for (auto x : {g.source, g.target, g.width, g.slots, g.sink}) program.word(x);
  program.word(g.families.size());
  for (auto [n, d] : g.families) {
    program.word(n); program.text(str(d, "key_id")); program.word(num(d, "scale"));
    program.vector(ints(d, "q")); program.vector(ints(d, "p"));
  }
  SmallVector<Attribute> receipts;
  std::vector<pass2::Exact> outputs;
  auto ins = field<ArrayAttr>(g.program, "instructions");
  program.word(g.edges.size());
  for (auto [i, e] : llvm::enumerate(g.edges)) {
    const auto kind = str(cast<DictionaryAttr>(ins[i]), "op");
    auto family = g.families.at(e.n);
    auto def = num(family, "scale");
    require(def > 0 && def <= 128, "unsupported CKKS default scale");
    auto prec = std::max<int64_t>(53, def);
    Hash payload;
    for (auto v : e.values) payload.word(std::bit_cast<uint64_t>(v));
    auto digest = payload.finish();
    program.text(e.id); program.text(kind);
    for (auto x : {e.n, e.level, e.k, e.shift, e.site}) program.word(x);
    program.word(std::bit_cast<uint64_t>(e.scale)); program.vector(e.inputs);
    program.word(e.values.size()); program.text(digest);
    pass2::Exact scale = e.inputs.empty() ? 1 : outputs.at(e.inputs[0]);
    if (e.op == Op::Source || !e.values.empty()) {
      double log = double(def);
      if (e.op != Op::Source) {
        const double parent = g.edges.at(e.inputs[0]).scale;
        log = e.op == Op::AddPlain ? parent : e.scale - parent;
      }
      const auto meta = pass2::plannedEncodingScale(log);
      const auto effective = pass2::effectiveEncoderScale(meta, prec);
      NamedAttrList row;
      row.set("index", b.getI64IntegerAttr(i)); row.set("node_id", b.getStringAttr(e.id));
      row.set("call", b.getStringAttr(kind)); row.set("key_id", family.get("key_id"));
      row.set("level", b.getI64IntegerAttr(e.level)); row.set("ring_n", b.getI64IntegerAttr(e.n));
      row.set("precision", b.getI64IntegerAttr(prec));
      row.set("branch", b.getStringAttr(prec == 53 ? "embedDouble/Scale.Float64" : "embedArbitrary/Scale.Value"));
      row.set("log_argument_exact", b.getStringAttr(pass2::exactBinary(log).get_str()));
      row.set("metadata_scale", b.getStringAttr(meta.get_str()));
      row.set("effective_encoder_scale", b.getStringAttr(effective.get_str()));
      row.set("payload_sha256", b.getStringAttr(digest));
      row.set("payload_count", b.getI64IntegerAttr(e.values.size()));
      e.encoding = row.getDictionary(b.getContext()); receipts.push_back(e.encoding);
      if (e.op == Op::Source) scale = meta;
      else if (e.op == Op::AddPlain) scale = std::max(scale, meta);
      else scale = pass2::roundCKKSScale(scale * meta);
    } else if (e.op == Op::Mul) scale = pass2::roundCKKSScale(scale * outputs.at(e.inputs[1]));
    else if (e.op == Op::Add) for (auto a : e.inputs) scale = std::max(scale, outputs.at(a));
    else if (e.op == Op::Rescale) {
      auto level = g.edges.at(e.inputs[0]).level;
      require(level - e.level == (def <= 64 ? 1 : 2), "rescale/precision mode mismatch");
      for (; level > e.level; --level)
        scale = pass2::roundCKKSScale(scale / pass2::Exact(std::to_string(ints(family, "q")[level])));
    }
    e.outputScale = scale.get_str(); outputs.push_back(std::move(scale));
  }
  NamedAttrList plan;
  plan.set("schema", b.getStringAttr("encoding-receipt-v1"));
  plan.set("abi", b.getStringAttr("lattigo/v6.2.0;batched-periodic-f64;mpfr-exp2-rn128;nearest-even"));
  plan.set("program_sha256", b.getStringAttr(program.finish()));
  plan.set("receipts", b.getArrayAttr(receipts));
  return plan.getDictionary(b.getContext());
}

Op opcode(StringRef s) {
  auto op = llvm::StringSwitch<std::optional<Op>>(s)
    .Case("source",Op::Source).Case("drop",Op::Drop).Case("mul",Op::Mul)
    .Case("mask",Op::Mask).Case("coefficient",Op::Coefficient)
    .Case("add_plain",Op::AddPlain).Case("add",Op::Add).Case("rescale",Op::Rescale)
    .Case("relin",Op::Relin).Case("baby",Op::Baby).Case("giant",Op::Giant)
    .Case("ring_down",Op::Down).Default(std::nullopt);
  require(op.has_value(), "unsupported_direct_opcode: " + s);
  return *op;
}
void periodic(DictionaryAttr d, int64_t n, int64_t width, int64_t slots,
              int64_t level, double scale) {
  require(num(d,"ring_dim")==n && num(d,"num_slots")==n/2 && num(d,"logical_width")==width &&
          num(d,"stride")==1 && num(d,"period")==slots && num(d,"lane_count")==1 &&
          str(d,"packing_mode")=="periodic_single" && ints(d,"active_lanes")==ArrayRef<int64_t>({0}) &&
          ints(d,"offsets")==ArrayRef<int64_t>({0}) && ints(d,"mask_signature").empty() &&
          num(d,"active_level")==level && real(d,"log2_scale")==scale,
          "layout_not_emittable_by_periodic_downconv");
}
} // namespace

EmitGraph::EmitGraph(DictionaryAttr p) : program(p),
    transition(p.getAs<DictionaryAttr>("ring_decision")), source(num(p,"source_ring")),
    target(num(p,"target_ring")), width(num(p,"width")), slots(num(p,"slots")), sink(num(p,"sink")) {
  require(width>0 && slots>=width && slots%width==0 && std::has_single_bit(uint64_t(slots)), "unsupported_encoded_layout");
  std::set<StringRef> keyIDs, ids;
  for (auto a: field<ArrayAttr>(p,"epochs")) {
    auto d=cast<DictionaryAttr>(a); auto n=num(d,"n");
    require((n==source || n==target) && std::has_single_bit(uint64_t(n)) && n/2>=slots &&
            !ints(d,"q").empty() && !ints(d,"p").empty() && num(d,"scale")>0 &&
            !str(d,"key_id").empty() && keyIDs.insert(str(d,"key_id")).second &&
            families.emplace(n,d).second,"invalid_concrete_key_domain");
  }
  require(families.contains(source) && families.contains(target) && (source!=target)==bool(transition),"missing_or_surplus_epoch");
  std::map<int64_t,std::set<int64_t>> keys;
  unsigned downs=0;
  for (auto a: field<ArrayAttr>(p,"instructions")) {
    auto d=cast<DictionaryAttr>(a); auto s=str(d,"ID");
    Edge e{opcode(str(d,"op")),s.str(),num(d,"n"),num(d,"Level"),num(d,"K"),
           real(d,"physical_log_scale"),ints(d,"inputs"),{}};
    auto kRing=num(d,"kernel_ring");
    require(!s.empty() && ids.insert(s).second && families.contains(e.n) && e.level>=0 &&
            size_t(e.level)<ints(families.at(e.n),"q").size() && std::isfinite(e.scale) && e.scale>0 &&
            e.k>=0 && e.k<=e.level, "invalid_edge_binding: " + s);
    bool sw=e.op==Op::Down, rot=e.op==Op::Baby || e.op==Op::Giant;
    require(kRing==(sw?source:e.n) && (e.k==0 || rot || e.op==Op::Relin),"invalid_kernel_domain_or_truncation: " + s);
    if (transition)
      require(str(d,"input_key_domain")==str(families.at(kRing),"key_id") &&
              str(d,"output_key_domain")==str(families.at(e.n),"key_id") &&
              str(d,"kernel_domain")== (sw?"switching_key_ext":kRing==source?"source":"target"),"inconsistent_direct_domain: " + s);
    for (int64_t i:e.inputs)
      require(i>=0 && size_t(i)<edges.size() && edges[i].n==kRing,"non_topological_or_cross_domain_operand: " + s);
    auto arity=e.inputs.size();
    bool sourceOp=e.op==Op::Source;
    require(sourceOp==edges.empty() && (sourceOp ? !arity : e.op==Op::Mul ? arity==2 : e.op==Op::Add ? arity>=2 : arity==1), "invalid_direct_arity: " + s);
    if (sourceOp) require(e.n==source && e.level==int64_t(ints(families.at(source),"q").size())-1 && e.scale==num(families.at(source),"scale"),"stale_input_binding");
    for (auto i:e.inputs) {
      int64_t gap=edges[i].level-e.level;
      require(e.op==Op::Drop ? gap>=0 : e.op==Op::Rescale ? gap==(num(families.at(e.n),"scale")<=64?1:2) : gap==0,"invalid_operand_level: " + s);
      if (e.op==Op::Add) require(std::abs(edges[i].scale-e.scale)<=1e-8,"unaligned physical add: " + s);
    }
    if (e.op==Op::Mask || e.op==Op::Coefficient || e.op==Op::AddPlain) {
      e.values=field<DenseF64ArrayAttr>(d,"values").asArrayRef();
      require(!e.values.empty() && slots%e.values.size()==0 && llvm::all_of(e.values,[](double x){return std::isfinite(x);}),"invalid_plaintext_payload: " + s);
    }
    if (e.op==Op::Relin) relin.insert(e.n);
    if (rot) {
      e.shift=num(d,"rotation"); e.site=num(d,"route_site");
      require(e.shift>0 && e.shift<e.n/2 && e.site>=0,"unbound_rotation: " + s);
      keys[e.n].insert(rotationElement(e.shift,std::countr_zero(uint64_t(e.n))));
      if (e.op==Op::Baby) {
        auto &group=hoists[{e.site,e.inputs.front()}];
        require(llvm::none_of(group,[&](size_t j){return edges[j].shift==e.shift;}),"duplicate_hoist_rotation");
        group.push_back(edges.size());
      }
    }
    if (sw) {
      require(transition && e.n==target && source>target && num(p,"cut_level")==e.level &&
              field<BoolAttr>(transition,"emittable").getValue() && str(transition,"op_kind")=="ring_down" && num(transition,"parallel_factor")==1 &&
              str(transition,"source_key")==str(families.at(source),"key_id") && str(transition,"target_key")==str(families.at(target),"key_id") &&
              num(transition,"switch_level_q")==e.level && num(transition,"switch_level_p")==int64_t(ints(families.at(target),"p").size())-1 &&
              ints(transition,"target_q")==ints(families.at(target),"q") && ints(transition,"target_p")==ints(families.at(target),"p") &&
              edges[e.inputs.front()].scale==e.scale,"non_emittable_conversion_or_key");
      auto cut=field<DictionaryAttr>(transition,"cut");
      auto values=field<ArrayAttr>(cut,"live_values");
      auto records=transition.getAs<ArrayAttr>("operands");
      require(values.size()>=1 && values.size()<=(records?2u:1u) && downs<values.size() &&
              num(cut,"ciphertext_use_count")==int64_t(values.size()) && (!records || records.size()==values.size()),"unconverted_live_cut_value");
      auto cutValue=cast<DictionaryAttr>(values[downs]);
      auto bind=records?cast<DictionaryAttr>(records[downs]):transition;
      auto provided=field<DictionaryAttr>(bind,"provided"), required=field<DictionaryAttr>(bind,"required");
      auto input=e.inputs.front(), raw=input;
      if (num(cutValue,"active_level")!=e.level) {
        require(edges[input].op==Op::Drop && edges[input].inputs.size()==1,"missing_cut_level_alignment");
        raw=edges[input].inputs.front();
      }
      periodic(provided,source,width,slots,e.level,e.scale);
      periodic(required,target,width,slots,e.level,e.scale);
      periodic(field<DictionaryAttr>(cutValue,"layout"),source,width,slots,edges[raw].level,e.scale);
      if(records) require(bind.get("value")==cutValue && str(bind,"input_edge")==edges[input].id &&
                          str(field<DictionaryAttr>(bind,"transition"),"ID")==s && !field<ArrayAttr>(bind,"consumers").empty(),"stale_cut_binding");
      auto uses=field<ArrayAttr>(cutValue,"uses");
      require(uses.size()==1 && str(cast<DictionaryAttr>(uses[0]),"role")== (downs?"bypass":"main") &&
              str(cast<DictionaryAttr>(uses[0]),"user")=="snabf.convert" && num(cast<DictionaryAttr>(uses[0]),"operand")==0 &&
              str(cutValue,"edge_id")==edges[raw].id && str(cutValue,"key_domain")==str(transition,"source_key") &&
              num(cutValue,"ring_dim")==source && num(cutValue,"active_level")==edges[raw].level &&
              real(cutValue,"physical_log_scale")==e.scale && edges[raw].scale==e.scale &&
              ints(cutValue,"q")==ints(families.at(source),"q").take_front(edges[raw].level+1) &&
              ints(cutValue,"p")==ints(transition,"target_p"),"stale_direct_cut_value");
      operands.push_back({cutValue,provided,required,edges[input].id,records?field<ArrayAttr>(bind,"consumers"):ArrayAttr{}});
      ++downs;
    }
    edges.push_back(e);
  }
  require(downs==(transition?field<ArrayAttr>(field<DictionaryAttr>(transition,"cut"),"live_values").size():0) &&
          sink>=0 && size_t(sink)<edges.size() && edges[sink].n==target,"invalid_epoch_sink");
  int64_t bytes=0;
  for (auto [n,d]:families) {
    require(llvm::equal(keys[n],ints(d,"galois")),"used_rotations_disagree_with_key_inventory");
    bytes+=int64_t(keys[n].size())*num(d,"key_bytes");
  }
  if (auto search=p.getAs<DictionaryAttr>("ring_search")) {
    auto domains=field<ArrayAttr>(search,"key_domains"); std::set<int64_t> accounted;
    if(transition) require(domains==transition.get("key_domains"),"stale_search_key_domains");
    for(auto a:domains) {
      auto d=cast<DictionaryAttr>(a); auto role=str(d,"role");
      if(role=="switching_key_ext") {
        require(transition && !switchingDomain && str(d,"basis")=="QP" && str(d,"key_kind")=="ring_down" &&
                field<BoolAttr>(d,"published").getValue() && num(d,"ring_dim")==source && num(d,"embedded_secret_ring")==target &&
                str(d,"source_key")==str(transition,"source_key") && str(d,"target_key")==str(transition,"target_key") &&
                ints(d,"q")==ints(transition,"target_q") && ints(d,"p")==ints(transition,"target_p") && num(d,"level_q")==num(transition,"switch_level_q") &&
                num(d,"level_p")==num(transition,"switch_level_p"),"switching_key_basis_not_bound");
        switchingDomain=d; continue;
      }
      auto n=num(d,"ring_dim");
      require(families.contains(n) && accounted.insert(n).second && role==(n==source?"source":"target"),"unaccounted_execution_domain");
      auto e=families.at(n);
      require(str(d,"key_id")==str(e,"key_id") && ints(d,"q")==ints(e,"q") && ints(d,"p")==ints(e,"p"),"stale_execution_domain_moduli");
      std::set<std::pair<StringRef,int64_t>> expected{{"secret",0}};
      if(n==source)expected.emplace("public",0);
      if(relin.contains(n))expected.emplace("relin",0);
      for(auto key:keys[n])expected.emplace("galois",key);
      for(auto k:field<ArrayAttr>(d,"keys")) {
        auto key=cast<DictionaryAttr>(k);auto kind=str(key,"role");
        require(str(key,"basis")=="QP" && field<BoolAttr>(key,"published").getValue()==(kind!="secret") &&
                expected.erase({kind,kind=="galois"?num(key,"galois"):0})==1,"stale_domain_key_inventory");
      }
      require(expected.empty(),"missing_domain_key");
    }
    require(accounted.size()==families.size() && bool(switchingDomain)==bool(transition),"missing_execution_domain");
  }
  require(!transition || (!operands.empty() && switchingDomain),"unbound_mixed_domain_or_cut");
  require(bytes==num(p,"galois_bytes"),"stale_emitted_key_bytes");
  encodingPlan = buildEncodingPlan(*this);
  if (auto old = p.get("encoding_receipts"))
    require(old == encodingPlan, "stale_encoding_scale_binding");
}

DictionaryAttr bindEncodingProgram(ModuleOp m, DictionaryAttr raw, bool firstCheck) {
  const auto fresh = EmitGraph(raw).encodingPlan;
  constexpr StringLiteral key = "snabf.encoding_receipts";
  if (auto old = m->getAttr(key)) require(old == fresh, "stale_encoding_scale_binding");
  else {
    require(firstCheck && !m->hasAttr("snabf.emit_sha256") && !m->hasAttr("snabf.ckks_direct") &&
            !m->hasAttr("snabf.ckks_epochs"), "missing_encoding_scale_binding");
    m->setAttr(key, fresh);
  }
  NamedAttrList bound(raw); bound.set("encoding_receipts", fresh);
  return bound.getDictionary(m.getContext());
}
} // namespace snabf::ir::lattigo_target::detail

#include "EmitGraph.h"
#include "pass1/SubgroupTower.h"
#include "llvm/ADT/StringSwitch.h"
#include <bit>
#include <cmath>

namespace snabf::ir::lattigo_target::detail {
namespace {
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
      require(e.op==Op::Drop ? gap>=0 : e.op==Op::Rescale ? gap==1 : gap==0,"invalid_operand_level: " + s);
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
}
} // namespace snabf::ir::lattigo_target::detail

#include "Pass2IR.h"
#include "pass2/NoiseTransfer.h"
#include <cmath>
#include <map>
#include <tuple>

namespace snabf::ir {
using namespace snabf::pass2;
namespace {
StringRef primitive(NoisePrimitive op) {
  switch(op){
  case NoisePrimitive::Source:return "source";case NoisePrimitive::Drop:return "drop";
  case NoisePrimitive::Mul:return "mul";case NoisePrimitive::Relin:return "relin";
  case NoisePrimitive::PtMul:return "pt_mul";case NoisePrimitive::Add:return "add";
  case NoisePrimitive::Rescale:return "rescale";case NoisePrimitive::AddPlain:return "add_plain";
  } throw ContractError("unknown primitive");
}
double real(DictionaryAttr d,StringRef k){return field<FloatAttr>(d,k).getValueAsDouble();}
int64_t integer(DictionaryAttr d,StringRef k){return field<IntegerAttr>(d,k).getInt();}
}
DictionaryAttr transferRecord(PolyEvalOp op,DictionaryAttr c,int64_t level) {
  Builder b(op.getContext()); auto ct=op.getInput().getType();
  auto profile=SymbolTable::lookupNearestSymbolFrom<ProfileOp>(op,ct.getProfile());
  auto domain=field<DenseF64ArrayAttr>(c,"input_interval"); auto id=field<StringAttr>(c,"candidate_id");
  NumericDAG dag;dag.output=integer(c,"output");
  for(auto a:field<ArrayAttr>(c,"nodes")){auto n=cast<DictionaryAttr>(a);PlanNode v;
    v.op=field<StringAttr>(n,"op").str();v.constant=real(n,"constant");v.depth=integer(n,"depth");
    auto w=field<DenseF64ArrayAttr>(n,"weights").asArrayRef();v.weights.assign(w.begin(),w.end());
    for(auto i:field<DenseI64ArrayAttr>(n,"args").asArrayRef())v.inputs.push_back(i);
    dag.nodes.push_back(std::move(v));}
  DictionaryAttr supplied; size_t site=0;
  for(auto &other:*op->getBlock()){if(&other==op.getOperation())break;site+=isa<PolyEvalOp>(other);}
  if(auto profiles=profile->getAttrOfType<ArrayAttr>("validated_poly_noise"))for(auto a:profiles){
    auto d=cast<DictionaryAttr>(a);
    if(integer(d,"site")==int64_t(site) && field<StringAttr>(d,"candidate_id")==id && integer(d,"input_level")==level){
      require(!supplied,"ambiguous arithmetic noise contract");supplied=d;}}
  ValidatedNoiseProfile p;using Key=std::tuple<int64_t,std::string,int64_t,int64_t,int64_t>;
  std::map<Key,DictionaryAttr> table;
  if(supplied){
    require(field<DenseI64ArrayAttr>(supplied,"q")==profile.getQAttr() && field<DenseI64ArrayAttr>(supplied,"p")==profile.getPAttr() &&
      field<StringAttr>(supplied,"key_id")==profile.getKeyIdAttr() && field<ArrayAttr>(supplied,"dag")==field<ArrayAttr>(c,"nodes"),
      "operation contract bound to another profile or DAG");
    p.contractId=field<StringAttr>(supplied,"contract_id").str();p.commonEvent=field<StringAttr>(supplied,"common_event").str();
    auto dom=field<DenseF64ArrayAttr>(supplied,"ideal_domain");require(dom.size()==2,"invalid validated domain");p.idealDomain=PlanRange{dom[0],dom[1]};
    p.validityFailure=real(supplied,"validity_failure");p.outputScale=real(supplied,"output_scale");
    require(*p.outputScale==std::ldexp(1.0,profile.getDefaultLogScale()),"contract output scale mismatch");
    for(auto a:field<ArrayAttr>(supplied,"sites")){auto d=cast<DictionaryAttr>(a);
      Key key{integer(d,"node"),field<StringAttr>(d,"primitive").str(),integer(d,"operand"),integer(d,"in_level"),integer(d,"out_level")};
      require(table.emplace(key,d).second,"duplicate primitive residual contract");}
    p.residual=[&](const NoiseSite &s)->std::optional<PrimitiveResidual>{auto it=table.find({s.node,primitive(s.primitive).str(),s.operand,s.inputLevel,s.outputLevel});
      if(it==table.end())return {};auto d=it->second;
      return PrimitiveResidual{field<StringAttr>(d,"common_event").str(), NoiseBounds::fromVarianceProxy(real(d,"mean_bias"),real(d,"variance_proxy"),real(d,"absolute")),real(d,"units_per_decoded")};};
  }
  auto t=evaluateNoiseTransfer(dag,{domain[0],domain[1]},level,real(c,"activation_error"),supplied?&p:nullptr);
  NamedAttrList r;SmallVector<Attribute> sites;
  for(auto &n:t.nodes)for(auto &s:n.sites){NamedAttrList d;d.set("node",b.getI64IntegerAttr(s.node));d.set("primitive",b.getStringAttr(primitive(s.primitive)));
    d.set("operand",b.getI64IntegerAttr(s.operand));d.set("in_level",b.getI64IntegerAttr(s.inputLevel));d.set("out_level",b.getI64IntegerAttr(s.outputLevel));sites.push_back(d.getDictionary(b.getContext()));}
  r.set("sites",b.getArrayAttr(sites));r.set("input_level",b.getI64IntegerAttr(level));r.set("delta_level",b.getI64IntegerAttr(t.cost.depth));
  r.set("output_level",b.getI64IntegerAttr(level-t.cost.depth)); r.set("output_interval",c.get("output_interval"));
  r.set("gain",c.get("gain"));r.set("bias_scale_ratio",b.getF64FloatAttr(1.0)); // materialized x and result both target DefaultScale
  r.set("canonical_approx_component",b.getF64FloatAttr(std::ldexp(real(c,"activation_error"),profile.getDefaultLogScale())));
  r.set("b_poly_components",b.getStringAttr("Delta_out * activation_error + canonical_evaluation_residual"));
  for(auto k:{"canonical_bias","variance_proxy","validity_failure"})r.set(k,b.getUnitAttr());
  r.set("status",b.getStringAttr(t.canonical?"validated":"symbolic"));
  if(t.canonical){auto &v=*t.canonical;r.set("canonical_bias",b.getF64FloatAttr(v.meanBias));r.set("variance_proxy",b.getF64FloatAttr(v.varianceProxy));
    r.set("validity_failure",b.getF64FloatAttr(v.validityFailure));r.set("contract_id",b.getStringAttr(v.contractId));}
  else r.set("unresolved",b.getStringAttr(t.reason));return r.getDictionary(b.getContext());
}
}

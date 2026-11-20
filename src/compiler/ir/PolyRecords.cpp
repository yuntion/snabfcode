#include "Pass2IR.h"
#include "pass2/CompositePolyDP.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace snabf::ir {
namespace {
using namespace snabf::pass2;
auto ints(Builder &b, const std::vector<unsigned> &x) {
  return b.getDenseI64ArrayAttr(SmallVector<int64_t>(x.begin(),x.end()));
}
auto range(Builder &b, PlanRange x) {return b.getDenseF64ArrayAttr({x.lo,x.hi});}
DictionaryAttr record(Builder &b, const PolyCandidate &c) {
  NamedAttrList d;
  auto i=[&](StringRef k,int64_t x){d.set(k,b.getI64IntegerAttr(x));};
  auto f=[&](StringRef k,double x){require(std::isfinite(x), "nonfinite certificate");d.set(k,b.getF64FloatAttr(x));};
  d.set("candidate_id",b.getStringAttr(c.id)); d.set("template",b.getStringAttr(c.templateName));
  i("delta_level",c.cost.depth); i("output",c.dag.output);
  i("ct_mults",c.cost.ctMults); i("pt_mults",c.cost.ptMults); i("relins",c.cost.relins);
  i("rescales",c.cost.rescales); i("adds",c.cost.additions); i("alignment_drops",c.cost.alignmentDrops);
  i("nominal_ps_mults",c.nominalPSMults); i("nominal_core_depth",c.nominalDepth);
  f("activation_error",c.approximationError); f("gain",c.outputGain);
  // This is the template's mathematical identity error, not its fitted residual.
  f("template_error",c.templateRefError); f("core_error",c.stages.back().errorOut);
  f("core_gain",c.coreGain); f("bypass_gain",c.bypassGain);
  d.set("input_interval",range(b,c.idealInput)); d.set("output_interval",range(b,c.actualOutput));
  SmallVector<Attribute> stages,nodes,work;
  for(auto &w:c.cost.byDepth){NamedAttrList v;
    v.set("mul",b.getI64IntegerAttr(w.ctProducts));v.set("square",b.getI64IntegerAttr(w.ctSquares));
    v.set("relin",b.getI64IntegerAttr(w.relins));v.set("pt_mul",b.getI64IntegerAttr(w.ptMults));v.set("rescale",b.getI64IntegerAttr(w.rescales));
    v.set("ct_add",b.getI64IntegerAttr(w.ctAdds));v.set("pt_add",b.getI64IntegerAttr(w.ptAdds));
    v.set("drops_by_span",ints(b,w.levelReductions));work.push_back(v.getDictionary(b.getContext()));}
  d.set("work_by_consumed_depth",b.getArrayAttr(work));
  for (auto &s:c.stages) {
    NamedAttrList a;
    a.set("reference",b.getStringAttr(s.stage)); a.set("schedule",b.getStringAttr(s.schedule));
    a.set("degree",b.getI64IntegerAttr(s.degree)); a.set("input_node",b.getI64IntegerAttr(s.inputNode));
    a.set("output_node",b.getI64IntegerAttr(s.outputNode)); a.set("coefficients",b.getDenseF64ArrayAttr(s.coefficients));
    a.set("i_in",range(b,s.idealIn)); a.set("a_in",range(b,s.actualIn)); a.set("j_in",range(b,s.hullIn));
    a.set("i_out",range(b,s.idealOut)); a.set("a_out",range(b,s.actualOut));
    a.set("epsilon",b.getF64FloatAttr(s.residual)); a.set("lip",b.getF64FloatAttr(s.lipschitz));
    a.set("e_in",b.getF64FloatAttr(s.errorIn)); a.set("e_out",b.getF64FloatAttr(s.errorOut));
    stages.push_back(a.getDictionary(b.getContext()));
  }
  require(c.noise.nodes.size()==c.dag.nodes.size(),"incomplete DAG contract");
  for (auto [idx,n]:llvm::enumerate(c.dag.nodes)) {
    auto &t=c.noise.nodes[idx]; NamedAttrList a;
    a.set("op",b.getStringAttr(n.op)); a.set("args",ints(b,n.inputs)); a.set("weights",b.getDenseF64ArrayAttr(n.weights));
    a.set("constant",b.getF64FloatAttr(n.constant)); a.set("depth",b.getI64IntegerAttr(n.depth));
    a.set("range",range(b,t.range)); a.set("operand_gains",b.getDenseF64ArrayAttr(t.operandGains));
    a.set("output_sensitivity",b.getF64FloatAttr(t.outputSensitivity));
    a.set("product_remainder",b.getBoolAttr(t.productRemainder));
    nodes.push_back(a.getDictionary(b.getContext()));
  }
  d.set("nodes",b.getArrayAttr(nodes)); d.set("stages",b.getArrayAttr(stages));
  d.set("noise_contract",b.getStringAttr("dependent-node-transfer-v1"));
  d.set("variance_proxy",b.getUnitAttr()); d.set("evaluation_bias",b.getUnitAttr());
  return d.getDictionary(b.getContext());
}
llvm::json::Value toJSON(Attribute a) {
  if (auto d=dyn_cast<DictionaryAttr>(a)) { llvm::json::Object out;
    for(auto item:d) out[item.getName().str()]=toJSON(item.getValue()); return out; }
  if (auto v=dyn_cast<ArrayAttr>(a)) {llvm::json::Array out; for(auto item:v) out.push_back(toJSON(item));return out;}
  if (auto v=dyn_cast<DenseI64ArrayAttr>(a)) {llvm::json::Array out; for(auto x:v.asArrayRef()) out.push_back(x); return out;}
  if (auto v=dyn_cast<DenseF64ArrayAttr>(a)) {llvm::json::Array out; for(auto x:v.asArrayRef()) out.push_back(x); return out;}
  if (auto v=dyn_cast<BoolAttr>(a)) return v.getValue();
  if (auto v=dyn_cast<IntegerAttr>(a)) return v.getInt();
  if (auto v=dyn_cast<FloatAttr>(a)) return v.getValueAsDouble();
  if (auto v=dyn_cast<StringAttr>(a)) return v.str();
  require(isa<UnitAttr>(a),"unsupported Pass 2 audit field"); return nullptr;
}
}
llvm::json::Object jsonPass2(DictionaryAttr d) {
  if (!d) return {}; auto v=toJSON(d); return std::move(*v.getAsObject());
}
DictionaryAttr pass2Manifest(ModuleOp m) {
  Builder b(m.getContext()); NamedAttrList out;
  if (!m->hasAttr("snabf.activation_sources")) return out.getDictionary(m.getContext());
  SmallVector<Attribute> sites,sinks;
  m.walk([&](PolyEvalOp op){ NamedAttrList d;
    d.set("input_level",b.getI64IntegerAttr(op.getInput().getType().getLevel()));
    d.set("output_level",b.getI64IntegerAttr(op.getOut().getType().getLevel()));
    d.set("selected",field<DictionaryAttr>(op,"poly_selected")); d.set("frontier",field<ArrayAttr>(op,"poly_frontier"));
    d.set("budget",field<BudgetAttr>(op,"budget").getFields()); sites.push_back(d.getDictionary(m.getContext())); });
  m.walk([&](SinkOp op){sinks.push_back(field<BudgetAttr>(op,"budget").getFields());});
  out.set("polynomials",b.getArrayAttr(sites)); out.set("sinks",b.getArrayAttr(sinks));
  out.set("decisions",field<ArrayAttr>(m,"snabf.poly_decisions"));
  out.set("joint_objective",field<FloatAttr>(m,"snabf.joint_objective"));
  return out.getDictionary(m.getContext());
}

void publishPolynomials(ModuleOp m,int64_t depth,double epsilon,int64_t minOut,double maxError,StringRef family) {
  require(m->hasAttr("snabf.activation_sources"), "Pass 2 needs source semantics: run snabf-mark-activations before HEIR");
  require(depth>=0 && depth<=INT32_MAX && std::isfinite(epsilon) && epsilon>=0 && minOut>=0 &&
      std::isfinite(maxError) && (maxError>=0 || maxError==-1), "invalid Pass 2 budget");
  Builder b(m.getContext());
  require(family=="all"||family=="direct"||family=="composed","invalid template family");
  m.walk([&](PolyEvalOp op){op->setAttr("depth_budget",b.getI64IntegerAttr(depth)); op->setAttr("approx_budget",b.getF64FloatAttr(epsilon));op->setAttr("allowed_templates",b.getStringAttr(family));});
  m.walk([&](SinkOp op){op->setAttr("min_output_level",b.getI64IntegerAttr(minOut));
    if(maxError>=0) op->setAttr("max_approx_error",b.getF64FloatAttr(maxError)); else op->removeAttr("max_approx_error"); });
  refreshPolynomials(m,false);
}
void refreshPolynomials(ModuleOp m,bool verify) {
  if (!m->hasAttr("snabf.activation_sources")) return;
  Builder b(m.getContext()); struct Edge { double bound; int64_t level; };
  llvm::DenseMap<Value,Edge> edges;
  m.walk([&](Operation *o){
    if(o->getNumResults()!=1 || !isa<CtType>(o->getResult(0).getType())) return;
    auto input=[&](unsigned i){require(edges.contains(o->getOperand(i)),"non-topological Pass 2 region");return edges.lookup(o->getOperand(i));};
    Edge out;
    if (auto s=dyn_cast<SourceOp>(o)) {
      auto profile=SymbolTable::lookupNearestSymbolFrom<ProfileOp>(o,s.getOut().getType().getProfile());
      out={field<FloatAttr>(s,"input_bound").getValueAsDouble(),int64_t(profile.getQ().size())-1};
    }
    else if(isa<ConvertOp>(o)){out=input(0);}
    else if(auto l=dyn_cast<LintransOp>(o)){out=input(0);out.bound=upperProduct(out.bound,linearNorm(preparedWeights(l.getTable(),l.getDiagonals()),true)); --out.level;}
    else if (auto p=dyn_cast<PolyEvalOp>(o)) {
      auto activation=field<StringAttr>(p,"activation").getValue();
      require(activation=="silu" || activation=="gelu", "unsupported activation contract");
      out=input(0); auto depth=field<IntegerAttr>(p,"depth_budget").getInt();
      auto eps=field<FloatAttr>(p,"approx_budget").getValueAsDouble();
      require(depth>=0 && depth<=INT32_MAX && std::isfinite(eps) && eps>=0 && out.level>=0 && out.bound>0, "invalid logical activation domain or budget");
      DPOptions options; options.inputLevel=out.level; options.maxDepth=std::min(depth,out.level);options.maxError=eps;
      auto family=field<StringAttr>(p,"allowed_templates").getValue();
      require(family=="all"||family=="direct"||family=="composed","invalid template admission policy");
      options.direct=family!="composed";options.composed=family!="direct";
      auto result=synthesizePolynomials(activation.str(),{-out.bound,out.bound},options); SmallVector<Attribute> records;
      unsigned least=UINT32_MAX; double largest=0;
      for(auto &c:result.frontier){auto raw=record(b,c); NamedAttrList fields(raw);
        fields.set("transfer",transferRecord(p,raw,out.level));records.push_back(fields.getDictionary(b.getContext()));
        least=std::min(least,c.cost.depth);largest=std::max({largest,-c.actualOutput.lo,c.actualOutput.hi});}
      require(!records.empty(),"empty " + activation + " Pareto frontier under full activation error and RNS depth budgets");
      auto frontier=b.getArrayAttr(records);
      if (verify) {
        require(p->getAttr("poly_frontier")==frontier,"stale Pass 2 frontier or certificate");
        require(llvm::is_contained(frontier,field<DictionaryAttr>(p,"poly_selected")),"stale selected full DAG");
      } else p->setAttr("poly_frontier",frontier);
      out.level-=least; out.bound=largest;
    } else if (auto s=dyn_cast<StepOp>(o); s && s.getKind()=="ckks.add") {
      auto a=input(0),c=input(1);out={upperSum(a.bound,c.bound),std::min(a.level,c.level)};
    } else throw ContractError("unsupported logical region while enumerating Pass 2");
    require(std::isfinite(out.bound),"nonfinite source interval"); edges[o->getResult(0)]=out;
  });
}
}

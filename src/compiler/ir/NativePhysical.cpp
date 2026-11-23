#include "NativePhysical.h"
#include "NativeReluReplay.h"
#include "pass3/ConcreteVerifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <array>
#include <bit>

namespace snabf::native {
namespace {
using R=ExactRational;
using T=ExactScaleTracker;
using O=pass3::Op;
using llvm::json::Array;
using llvm::json::Object;
struct Wire { std::string id; ScaleState s; R mag, err; };
void need(bool ok,const char *why) { if(!ok) throw std::invalid_argument(why); }
R rat(double f){return pass2::exactBinary(f);}
R integer(uint64_t v){return R(std::to_string(v));}
template<class A> Array ints(const A &v){Array a;for(auto x:v)a.push_back(int64_t(x));return a;}
Object encoding(const EncodingCallReceipt &pt, std::span<const double> payload) {
  Array bits;
  for(double v:payload) {
    std::string s=llvm::utohexstr(std::bit_cast<uint64_t>(v),true);
    bits.push_back(std::string(16-s.size(),'0')+s);
  }
  return Object{{"meta",pt.realized_meta_scale.get_str()},{"effective",pt.realized_eff_scale.get_str()},
    {"requested",pt.requested_scale.get_str()},{"call",pt.call_id},{"branch",pt.branch},{"precision",pt.encoder_prec_bits},
    {"bits",std::move(bits)},{"sha",pt.argument_sha256.value_or("")}};
}
} // namespace

Object nativeReluWitness(const NativeArithmeticProgram &p) {
  auto bits=[](double a){auto s=llvm::utohexstr(std::bit_cast<uint64_t>(a),true);return std::string(16-s.size(),'0')+s;};
  Array sites;
  for(size_t i=0;i<p.polynomials.size();++i) {
    const auto &c=p.polynomials[i]; if(!c.relu)continue;
    const auto &site=p.activationSites.at(i); const auto &proof=*c.relu;
    Array stages,dag,slots; const Object *lock=nullptr;
    for(auto &l:*p.source.inspection.getArray("activation_locks"))
      if(l.getAsObject()->getString("site")==site)lock=l.getAsObject();
    need(lock && lock->getString("kind")=="relu","missing original ReLU lock");
    for(auto a:p.reluSlotDomains.at(site))slots.push_back(Array{a.lo,a.hi});
    for(const auto &s:c.stages){Array co;for(double x:s.coefficients)co.push_back(bits(x));
      stages.push_back(Object{{"input",s.inputNode},{"output",s.outputNode},{"degree",s.degree},{"bits",std::move(co)}});}
    for(const auto &n:c.dag.nodes){Array w;for(double x:n.weights)w.push_back(bits(x));
      dag.push_back(Object{{"op",n.op},{"inputs",ints(n.inputs)},{"weights",std::move(w)},{"constant",bits(n.constant)},{"depth",n.depth}});}
    sites.push_back(Object{{"source_lock",Object(*lock)},{"template",c.templateName},{"site",site},
      {"affine_slots",std::move(slots)},{"axis",Array{c.idealInput.lo,c.idealInput.hi}},
      {"mpfi160",Array{proof.mpfi[0],proof.mpfi[1]}},{"bernstein",Array{proof.bernstein[0],proof.bernstein[1]}},
      {"cells",ints(proof.cells)},{"bernstein_budget",proof.bernsteinBudget},{"stages",std::move(stages)},
      {"dag",std::move(dag)},{"output",c.dag.output}});
  }
  return Object{{"schema","snabf.native-relu-witness.v1"},{"source",p.source.inspection.getString("source_sha256")->str()},
    {"input_spec",p.source.inspection.getString("input_spec_sha256")->str()},{"sites",std::move(sites)}};
}

void checkNativeReluWitness(const NativeArithmeticProgram &p,const Object *claim) {
  need(!p.reluSlotDomains.empty() && claim && *claim==nativeReluWitness(p),
       "[SNABF_NATIVE_RELU_WITNESS] stale source, affine slots, odd chain, reconstruction or proof");
}

R nativeBlockSwitchError(const ScaleDomain &d,int level) {
  d.check(level);
  need(d.p.size()<=2,"unsupported native special basis");
  R P=1, sum=0;
  for(auto x:d.p) P*=integer(x);
  const int width=d.p.size();
  for(int b=0;b<=level;b+=width) {
    R digit=1;
    for(int l=b;l<std::min(level+1,b+width);++l) digit*=integer(d.q[l]);
    sum+=digit+1;
  }
  return R(10)*d.ring_n*d.ring_n*sum/P;
}

Object lowerNativePhysical(const NativeArithmeticProgram &p,const pass3::Solution &input,const pass3::Options &opt,const R &budget,Object *audit,pass3::Solution *verified) {
  const auto &g=p.graph; auto s=input;
  auto replay=pass3::replay(g,opt,s);
  const bool relu=!p.reluSpans.empty();
  const bool n2=!relu && opt.alpha>0;
  if(relu) need(g.n==65536 && g.levels==29 && s.p.size()==2 && opt.alpha==2 && opt.maxK==0 &&
    opt.securityCap==1400 && opt.tolerance==0x1p-12 && p.reluSpans.size()==p.reluSlotDomains.size() &&
    std::all_of(g.edges.begin(),g.edges.end(),[&](auto &e){return !e.ring || e.ring==g.n;}),"wrong native ReLU profile");
  // The scalar replay validates the concrete basis and scales before its
  // triangle-bound arithmetic. Only that latter model can be superseded,
  // starting at the first bound ReLU site. Every physical edge is rechecked.
  bool replace=false;
  if(relu && s.physical.size()==g.edges.size()) {
    if(s.errors.size()>=size_t(p.reluSpans.front().first) && s.errors.size()<g.edges.size())
      replace=replay.reason=="arithmetic allowance exceeded at "+g.edges[s.errors.size()].id;
    if(s.errors.size()>size_t(p.reluSpans.front().first))
      replace|=replay.reason=="concrete message/scale headroom at "+g.edges[s.errors.size()-1].id;
    replace|=replay.reason=="sink approximation + arithmetic bound exceeds target";
  }
  if(!replay.reason.empty()&&!replace) throw std::invalid_argument("unreplayable candidate: "+replay.reason);
  need((s.p.size()==1 || (s.p.size()==2 && opt.alpha==2)) &&
       std::all_of(s.k.begin(),s.k.end(),[](int k){return k==0;}),"unsupported native special-basis/truncation profile");
  need(!n2 || size_t(opt.alpha)==s.p.size(),"N2 alpha does not match the concrete P basis");
  need(s.scale<=53 && budget>=0 && opt.guardBits==1,"unsupported encoding/guard policy");
  ScaleDomain d{"native.source",g.n,s.scale,{s.q.begin(),s.q.end()},{s.p.begin(),s.p.end()}};
  d.check(g.levels-1);
  const int slots=p.source.domain.encodedSlots;
  R P=1; for(auto prime:d.p)P*=integer(prime);
  const R N=g.n, S=uint64_t(std::ceil(pass3::secretBound(g.n)));
  const R round=N*(1+S)/2;
  std::vector<R> modulus; R qprod=1;
  for(auto q:d.q) {qprod*=integer(q); modulus.push_back(qprod);}
  if(relu) need(qprod*P<=R(mpz_class(1)<<1400),"concrete QP outside exact ReLU envelope");
  std::map<int,R> switchTerms;
  std::optional<NativeReluReplay> scope;size_t span=0;Array proofs;
  Array records;std::vector<Wire> wires;
  for(size_t i=0;i<g.edges.size();++i) {
    const auto &e=g.edges[i]; Wire w{e.id,{0,0},rat(e.magnitude),0};
    if(span<p.reluSpans.size() && int(i)==p.reluSpans[span].first){
      auto &rs=p.reluSpans[span];auto &v=wires.at(rs.entry);
      scope.emplace(rs,v.id,v.err+abs(v.s.effective/v.s.metadata-1)*v.mag);
    }
    Object rec{{"id",e.id},{"op",pass3::name(e.op)},{"level",e.level},{"inputs",ints(e.inputs)},{"shift",e.shift}};
    auto wire=[&](size_t j)->const Wire&{return wires.at(e.inputs.at(j));};
    auto plain=[&](const R &requested,bool dyn){
      EncodingCallContext call;call.call_id="pt:"+e.id;call.site_id=e.id;call.operand_index=dyn?0:1;
      call.level=e.level;call.slots=slots;call.domain=d;call.is_batched=true;call.arg_type="[]float64";
      call.arg_precision_bits=53;call.encoder_prec_bits=53;call.dynamic_input=dyn;
      const std::span<const double> words=dyn?std::span<const double>{}:p.publicOperands.at(i);
      auto receipt=T::encode(call,requested,words);
      rec["plain"]=encoding(receipt,words);
      return ScaleState{receipt.realized_meta_scale,receipt.realized_eff_scale};
    };
    Array joins;
    auto join=[&](const Wire &a,const Wire &b,const std::string &id){
      auto side=[&](const Wire &v){return NativeScaleOperand{{v.id,d,e.level,slots,v.s},v.mag};};
      NativeScaleAddRequest r{id,side(a),side(b),budget,e.level};
      auto checkedJoin=[&](){try{return lowerNativeScaleAdd(r);}catch(const std::exception &failure){
        if(!relu)throw;
        auto j=T::analyzeAddSubMismatch(JoinOp::Add,a.s,b.s,a.mag,b.mag);
        std::string why="[SNABF_NATIVE_RELU_JOIN] "+id+"; bias="+(j.decoded_bias?j.decoded_bias->get_str():"missing")+"; budget="+budget.get_str()+"; "+failure.what();
        if(audit)(*audit)["first_join_refusal"]=Object{{"id",id},{"lhs_meta",a.s.metadata.get_str()},{"rhs_meta",b.s.metadata.get_str()},
          {"lhs_eff",a.s.effective.get_str()},{"rhs_eff",b.s.effective.get_str()},{"bias",j.decoded_bias?j.decoded_bias->get_str():"missing"},{"budget",budget.get_str()}};
        throw std::invalid_argument(why);
      }};
      auto v=checkedJoin(); auto *plan=std::get_if<NativeScaleAddPlan>(&v);
      need(plan && plan->preparation.empty(),"physical specialization needs an upstream re-solve");
      auto &j=plan->add.scales; const R M=j.out_meta_scale;
      if(scope)scope->join(id,a.id,R(j.lhs_int_mul)*a.s.metadata/M,b.id,R(j.rhs_int_mul)*b.s.metadata/M,M,e.level);
      const R inherited=(R(j.lhs_int_mul)*a.s.metadata*a.err+R(j.rhs_int_mul)*b.s.metadata*b.err)/M;
      joins.push_back(Object{{"lhs",a.id},{"rhs",b.id},{"lhs_mul",j.lhs_int_mul},{"rhs_mul",j.rhs_int_mul},
        {"bias",plan->charged_decoded_bias.get_str()},{"meta",M.get_str()}});
      // Mixed effective gains cannot be relabelled as a homogeneous sum.
      // Rebase the reference only AFTER charging both operands' mismatch.
      return Wire{id,{M,M},a.mag+b.mag,inherited+plan->charged_decoded_bias};
    };
    switch(e.op) {
      case O::Source: {
        w.s=plain(R(mpz_class(1)<<s.scale),true);
        w.err=(round+N+20*N*(N+1+S)/P)/w.s.metadata;break;
      }
      case O::Mask:case O::Coefficient: {
        auto &a=wire(0);
        R requested=e.op==O::Mask?integer(d.q.at(e.maskLevel)):
          pass2::plannedEncodingScale(s.physical[i]-s.physical.at(e.inputs[0]));
        if(scope && e.op==O::Coefficient){
          if(scope->targets.empty())scope->targets=NativeReluReplay::scaleTargets(g.polys.at(e.poly),s);
          requested=scope->targets.at(e.node)*integer(d.q.at(e.level))/a.s.metadata;
        }
        auto pt=plain(requested,false);w.s=T::multiply(a.s,pt);
        if(scope){need(e.op==O::Coefficient && p.publicOperands.at(i).size()==1,"non-scalar ReLU operand");
          auto k=scope->encoded(rat(p.publicOperands.at(i)[0]),a.s.metadata*pt.effective/w.s.metadata,
                                N*a.s.metadata/w.s.metadata,e.id);
          scope->ids[e.id]=scope->circuit.mul(scope->ids.at(a.id),k);break;}
        const R weight=rat(e.gains.at(0));
        w.err=(a.err*a.s.metadata*pt.effective*weight+(a.s.effective*a.mag+a.err*a.s.metadata)*N)/w.s.metadata;
        break;
      }
      case O::Multiply: {
        auto &a=wire(0),&b=wire(1);w.s=T::multiply(a.s,b.s);
        if(scope){auto prod=scope->circuit.mul(scope->ids.at(a.id),scope->ids.at(b.id));
          scope->ids[e.id]=scope->factor(prod,a.s.metadata*b.s.metadata/w.s.metadata,e.id);break;}
        R ea=a.s.metadata*a.err, eb=b.s.metadata*b.err;
        w.err=(ea*b.s.effective*b.mag+eb*a.s.effective*a.mag+ea*eb)/w.s.metadata;break;
      }
      case O::AddPlain: {
        auto &a=wire(0); auto pt=plain(a.s.metadata,false);
        Wire addend{"pt:"+e.id,pt,rat(std::abs(p.publicOperands.at(i).at(0))),N/pt.metadata};
        if(scope)scope->ids[addend.id]=scope->encoded(rat(p.publicOperands.at(i).at(0)),pt.effective/pt.metadata,N/pt.metadata,e.id);
        auto y=join(a,addend,e.id);w.s=y.s;w.err=y.err;break;
      }
      case O::Add: {
        Wire acc=wire(0);need(e.inputs.size()>=2,"invalid add arity");
        for(size_t j=1;j<e.inputs.size();++j)
          acc=join(acc,wire(j),j+1==e.inputs.size()?e.id:e.id+".partial"+std::to_string(j));
        w.s=acc.s; w.err=acc.err;break;
      }
      case O::Baby: case O::Giant:case O::Relin: {
        auto &a=wire(0);w.s=a.s;
        if(!switchTerms.contains(e.level)) switchTerms.emplace(e.level,nativeBlockSwitchError(d,e.level));
        if(n2) rec["ks_before_round"]=switchTerms.at(e.level).get_str();
        if(scope){need(e.op==O::Relin,"non-scalar key switch inside ReLU");
          auto key=scope->fresh(scope->ids.at(a.id),switchTerms.at(e.level)/w.s.metadata,e.id+":block-ks");
          scope->ids[e.id]=scope->fresh(key,round/w.s.metadata,e.id+":round");break;}
        w.err=a.err+(switchTerms.at(e.level)+round)/w.s.metadata;break;
      }
      case O::Rescale: {
        auto &a=wire(0);auto level=g.edges[e.inputs[0]].level;
        std::array<uint64_t,1> tail{d.q.at(level)};
        auto rs=T::rescaleForLowering(a.s,d,level,tail,1);
        need(rs.out_level==e.level,"non-unit rescale"); w.s={rs.meta_after,rs.eff_after};
        rec["dropped_q"]=ints(tail);
        if(scope){auto v=scope->factor(scope->ids.at(a.id),a.s.metadata/(integer(tail[0])*w.s.metadata),e.id);
          scope->ids[e.id]=scope->fresh(v,round/w.s.metadata,e.id+":round");break;}
        w.err=(a.err*a.s.metadata/integer(tail[0])+round)/w.s.metadata;break;
      }
      case O::Drop: {auto &a=wire(0);need(e.level<g.edges[e.inputs[0]].level,"non-decreasing drop");w.s=a.s;w.err=a.err;
        if(scope)scope->ids[e.id]=scope->ids.at(a.id);break;}
      default:throw std::invalid_argument("unsupported native physical operation");
    }
    // Same engineering primitive allowances as concrete Pass 3, but in the
    // actual scale units. Positive GMP products/sums retain all correlations
    // conservatively. This is not an issued conditional noise certificate.
    if(!scope){need(2*(w.s.effective*w.mag+w.s.metadata*w.err)<modulus.at(e.level),"physical headroom exhausted");
      if(relu)need(w.err<=rat(8*opt.tolerance) && 2*w.s.metadata<modulus.at(e.level),"native scalar allowance/headroom outside ReLU");}
    if(!joins.empty())rec["joins"]=std::move(joins);
    rec["meta"]=w.s.metadata.get_str();rec["effective_reference"]=w.s.effective.get_str();
    records.push_back(std::move(rec));wires.push_back(std::move(w));
    if(!scope)continue;
    scope->finishEdge(i,e.id);
    if(int(i)!=scope->span.output)continue;
    R lo=0,hi=0;for(const auto &a:p.reluSlotDomains.at(scope->span.site)){lo=std::min(lo,rat(a.lo));hi=std::max(hi,rat(a.hi));}
    if(audit){Array witness,nodes,endpoints;
      for(const auto &n:scope->circuit.nodes){const char *names[]={"input","literal","add","mul","anchor"};
        Object row{{"op",names[int(n.op)]},{"a",int64_t(n.a)},{"b",int64_t(n.b)},
          {"v",n.value.get_str()},{"rho",n.radius.get_str()},{"label",n.name}};
        if(n.fixedDelta)row["fixed"]=n.fixedDelta->get_str();nodes.push_back(std::move(row));}
      (*audit)["circuit"]=std::move(nodes);
      for(const R &x:{lo,hi}){for(int sign:{-1,1}){try{auto w=pass2::relu_jacobian::pointWitness(scope->circuit,x,sign);
        witness.push_back(Object{{"x",x.get_str()},{"signs",ints(w.signs)},{"decoded_delta",Array{w.diff.lo,w.diff.hi}}});}
        catch(const std::overflow_error&){witness.push_back(Object{{"x",x.get_str()},{"nonfinite",true}});}}
        try{auto b=pass2::relu_jacobian::cell(scope->circuit,pass2::relu_jacobian::enclose(x,x));
          endpoints.push_back(Object{{"x",x.get_str()},{"upper",b.back().value.r}});}
        catch(const std::overflow_error&){endpoints.push_back(Object{{"x",x.get_str()},{"nonfinite",true}});}}
      (*audit)["endpoint_enclosures"]=std::move(endpoints);
      (*audit)["box_point_witnesses"]=std::move(witness);}
    const bool fine=s.scale==46 && s.p==nativeReluBasis(p,6).p && s.q==nativeReluBasis(p,6).q;
    auto proof=pass2::relu_jacobian::prove(scope->circuit,lo,hi,4,10,fine?10:0,opt.tolerance/6);
    Array sources,edges,joinsChecked;
    for(size_t j=0;j<scope->circuit.nodes.size();++j){auto &n=scope->circuit.nodes[j];if(n.radius==0)continue;
      auto gain=proof.outputGain.find(j);sources.push_back(Object{{"id",n.name},{"radius",n.radius.get_str()},
        {"output_abs_partial",gain==proof.outputGain.end()?0:gain->second}});}
    if(audit){Array landmarks;
      for(int k:scope->span.stageEnds){auto id=scope->edges.at(k-scope->span.first);
        landmarks.push_back(Object{{"edge",g.edges[k].id},{"error",proof.error.at(id)},{"magnitude",proof.magnitude.at(id)}});}
      (*audit)["enclosure"]=Object{{"stages",std::move(landmarks)},{"cells",int64_t(proof.cells)},
        {"output_error",proof.error.at(scope->edges.back())},{"sources",Array(sources)}};}
    auto headroom=[&](int level,const R &m,size_t id,const std::string &name){
      const R used=2*m*std::max(R(1),rat(proof.magnitude.at(id)));
      if(used>=modulus.at(level))throw std::invalid_argument("[SNABF_NATIVE_RELU_WRAP] "+name+"; twice_bound="+used.get_str()+"; Q="+modulus[level].get_str());};
    for(size_t j=0;j<scope->edges.size();++j){size_t ix=scope->span.first+j,id=scope->edges[j];auto &v=wires.at(ix);
      headroom(g.edges[ix].level,v.s.metadata,id,v.id);
      // Rebase from raw/M - ideal to the suffix convention raw/M - E/M*ideal.
      v.err=rat(proof.error.at(id))+abs(v.s.effective/v.s.metadata-1)*v.mag;
      edges.push_back(Object{{"edge",v.id},{"magnitude",proof.magnitude[id]},{"decoded_error",proof.error[id]}});
    }
    for(const auto &v:scope->partials){headroom(v.level,v.scale,v.value,v.name);
      joinsChecked.push_back(Object{{"id",v.name},{"level",v.level},{"magnitude",proof.magnitude.at(v.value)}});}
    proofs.push_back(Object{{"site",scope->span.site},{"model","full-complex-disk-jacobian-mpfi160-bernstein-v1"},
      {"axis",Array{lo.get_str(),hi.get_str()}},{"cells",int64_t(proof.cells)},{"sources",std::move(sources)},
      {"edges",std::move(edges)},{"join_intermediates",std::move(joinsChecked)},{"output_error",proof.error.at(scope->edges.back())}});
    if(audit)(*audit)["spans"]=Array(proofs);
    scope.reset();++span;
  }
  need(!scope && span==p.reluSpans.size(),"unfinished ReLU physical span");
  auto &v=wires.at(g.sink);
  const R total=v.err+abs(v.s.effective/v.s.metadata-1)*v.mag+rat(g.approximation);
  if(audit){(*audit)["sink_upper"]=pass2::upperDouble(total);(*audit)["tolerance"]=opt.tolerance;}
  if(relu && total>rat(opt.tolerance))throw std::invalid_argument(llvm::formatv("[SNABF_NATIVE_RELU_SINK] upper={0:e8}; limit={1:e8}",pass2::upperDouble(total),opt.tolerance).str());
  need(total<=rat(opt.tolerance),"exact-physical engineering allowance exceeded at sink");
  auto &port=*p.source.inspection.getArray("inputs")->front().getAsObject();
  Object result{{"schema","snabf.native-single-physical.v1"},{"source",p.source.inspection.getString("source_sha256")->str()},
    {"input_spec",p.source.inspection.getString("input_spec_sha256")->str()},{"ring",g.n},{"scale",s.scale},
    {"q",ints(d.q)},{"p",ints(d.p)},{"width",int64_t(p.source.width)},{"slots",slots},{"sink",g.sink},
    {"edges",std::move(records)},{"input_domain",Array{*(*port.getArray("domain"))[0].getAsNumber(),*(*port.getArray("domain"))[1].getAsNumber()}},{"local_join_budget",budget.get_str()},
    {"engineering_upper",pass2::upperDouble(total)},{"probabilistic_cert_issued",false}};
  if(relu){result["relu_witness"]=nativeReluWitness(p);result["relu_jacobian"]=std::move(proofs);}
  if(n2) result["n2_search"]=Object{{"alpha",opt.alpha},{"scale_mode",opt.heterogeneous?"wide":"narrow"},
    {"min_scale",opt.minScale},{"max_scale",opt.maxScale},{"max_k",opt.maxK},{"cap_bits",int(opt.securityCap)},
    {"heir_scale_anchor",opt.heirScaleAnchor},{"prime_bw",Array{opt.primeWidthMin,opt.primeWidthMax}},
    {"aux_margin_bits",opt.auxMarginBits}};
  auto canonical=llvm::formatv("{0}",llvm::json::Value(Object(result))).str();
  result["emit_sha256"]=llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(canonical)),true);
  if(verified){s.errors.clear();for(const auto &w:wires)s.errors.push_back(pass2::upperDouble(w.err+abs(w.s.effective/w.s.metadata-1)*w.mag));*verified=std::move(s);}
  return result;
}
} // namespace snabf::native

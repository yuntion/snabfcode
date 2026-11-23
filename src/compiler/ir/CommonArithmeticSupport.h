#pragma once
#include "NoiseCertVerifier.h"

namespace snabf::ir::noise {
// Induction on the concrete centered decryption, not on either ciphertext
// component. Uniform over all embeddings, including repeated sparse slots.
inline void commonArithmeticSupport(const Topology &g, size_t i, std::vector<LocalStep> &steps,
    const std::optional<Exact> &scale, const std::optional<Exact> &ideal,
    const std::optional<Exact> &drift, const Exact &ma, const Exact &mb) {
  if (!g.fresh_width || g.domains.size()!=1 || !scale || !ideal || !drift) return;
  const Node &n=g.nodes.at(i); const Domain &d=g.domains.at(n.domain);
  auto &out=steps.at(i);
  const Exact N=d.ring_n, I=*scale * *ideal;
  const Exact kappa(46341,65536); // >= 1/sqrt(2), all odd-root real projections
  auto word=[](uint64_t x){ return Exact(std::to_string(x)); };
  auto floor=[](const Exact &x)->Exact { return mpz_class(x.get_num()/x.get_den()); }; // nonnegative
  Exact Q=1; for (int l=0;l<=n.level;++l) Q*=word(d.q.at(l));
  auto row=[&](Role role)->LocalTransfer& {
    const auto id="node/"+n.id+"/"+name(role);
    for(auto &r:out.rows) if(r.site==id) return r;
    auto &r=out.rows.emplace_back();r.site=id;r.active_q=Q;
    for(auto j:n.inputs) r.inputs.push_back(g.nodes[j].id);
    return r;
  };
  std::map<std::string,std::string> facts{{"key_id",d.key_id},{"source",g.nodes.front().id},
      {"target","fresh-v1/all-embeddings"},{"ideal_units",I.get_str()}};
  auto put=[&](const char* k,const Exact &v){facts[k]=v.get_str();};
  if(n.op==Op::Source) {
    if(out.fresh_coeff && 2 * *out.fresh_coeff >= Q-1) return;
    if(!out.fresh_coeff || !out.fresh_slot_error) return; // only the reconstructed, full PK call can seed this
    const Exact P=word(d.p.at(0));
    const Exact pk=floor((19*(2*N+1)+(N+1)*(P-1)/2)/P);
    const Exact error=*out.fresh_slot_error-(1-kappa)*N*pk;
    out.arithmetic=ArithmeticSupport{1,*out.fresh_coeff,I,error};
    put("degree",1);put("coeff",*out.fresh_coeff);put("error",error);
    put("real_projection_upper",kappa);
    row(Role::Encrypt).centered_support=std::move(facts);return; // not a fresh conditional MGF
  }
  if(n.inputs.empty()) return;
  const auto &oa=steps.at(n.inputs[0]).arithmetic;
  if(!oa) return;
  const auto &a=*oa;put("lhs_coeff",a.coeff);put("lhs_error",a.error);put("lhs_degree",a.degree);
  const Exact A=a.ideal_units;
  Exact raw=0,error=0; int degree=a.degree;
  struct Local { Role role; Exact bias; std::optional<Exact> lhs,rhs; };
  std::vector<Local> effects;
  auto add=[&](Role role,const Exact &b,std::optional<Exact> left={},std::optional<Exact> right={}) {
    effects.push_back({role,b,left,right});
  };
  switch(n.op) {
  case Op::CtMul: case Op::Add: case Op::Sub: {
    const auto &ob=steps.at(n.inputs.at(1)).arithmetic;
    if(!ob || a.degree!=1 || ob->degree!=1) return;
    const auto &b=*ob;put("rhs_coeff",b.coeff);put("rhs_error",b.error);put("rhs_degree",b.degree);
    if(n.op==Op::CtMul) {
      degree=2;raw=N*a.coeff*b.coeff;
      const Exact le=a.error*b.ideal_units,re=b.error*A,cross=a.error*b.error;
      put("left_term",le);put("right_term",re);put("cross",cross);
      error=le+re+cross;add(Role::Product,error);
    } else {
      const Exact signedB=n.op==Op::Sub ? Exact(-mb) : mb;
      if(n.inputs[0]==n.inputs[1]) {raw=abs(ma+signedB)*a.coeff;error=abs(ma+signedB)*a.error;}
      else {raw=abs(ma)*a.coeff+abs(mb)*b.coeff;error=abs(ma)*a.error+abs(mb)*b.error;}
      add(n.op==Op::Sub?Role::Sub:Role::Add,0,ma,signedB);
    }
    break;
  }
  case Op::Relin: case Op::Rotate: {
    const bool rel=n.op==Op::Relin;
    if(a.degree!=(rel?2:1) || n.k<0 || n.k>1 || (rel && n.k!=1) || d.p.size()!=1 ||
       n.evk!=(rel?d.key_id+":relin":d.key_id+":rotation:"+std::to_string(n.rotation))) return;
    degree=1;const Exact P=word(d.p[0]); Exact sum=0;
    for(int l=n.k;l<=n.level;++l) sum+=word(d.q[l]);
    const Exact D=n.k?word(d.q[0]):Exact(1);
    const Exact sw=floor((19*N*sum+(N+1)*(P-1)/2)/P);
    const Exact cut=(rel?Exact(N*N):N)*(D-1)/2;
    const Exact switchSlot=kappa*N*sw;
    const Exact cutSlot=(rel?Exact(kappa*N*N*N):Exact(N*N))*(D-1)/4;
    raw=a.coeff+sw+cut;error=a.error+switchSlot+cutSlot;
    put("switch_slot",switchSlot);put("cut_slot",cutSlot);
    put("P",P);put("prefix_product",D);put("digit_sum",sum);put("switch_coeff",sw);put("cut_coeff",cut);
    facts["evk"]=n.evk;
    if(!rel) add(Role::Permute,a.error); // complete transformed parent, not a signed token
    add(Role::KeySwitch,switchSlot);if(n.k) add(Role::GadgetTrunc,cutSlot);
    break;
  }
  case Op::Rescale: {
    const int in=g.nodes[n.inputs[0]].level;
    if(a.degree!=1 || d.rescale_primes!=1 || in!=n.level+1) return;
    const Exact p=word(d.q[in]), D=(N+1)*(p-1)/2;
    const Exact residual=(kappa*N+N*N/2)*(p-1)/(2*p);
    raw=(a.coeff+D)/p;error=a.error/p+residual;
    put("dropped_q",p);put("same_key_remainder",D);put("round_slot",residual);
    add(Role::RescaleRound,residual,Exact(1)/p);
    break;
  }
  case Op::Neg: case Op::Drop:
    if(a.degree!=1) return;
    raw=a.coeff;error=a.error;add(n.op==Op::Neg?Role::Neg:Role::Drop,0,n.op==Op::Neg?-1:1);break;
  case Op::ScalarMulPlain: case Op::VectorMulPlain: case Op::AddPlain: {
    auto &enc=row(Role::PlainEncode);
    if(a.degree!=1 || enc.formula.empty() || !enc.bias) return;
    if(enc.diagonal && n.op==Op::VectorMulPlain) {
      const auto &v=*enc.diagonal;const Exact L=2*v.period*v.coeff_abs;
      const Exact M=std::min(L,Exact(*n.effective_encoder_scale*v.weight_abs+v.delta_inf));
      raw=L*a.coeff;error=M*a.error+*enc.bias;put("plain_l1",L);put("plain_slot_abs",M);
      add(Role::PlainMul,M*a.error);
    } else if(enc.encoded_constant) {
      const Exact c=*enc.encoded_constant;
      if(n.op==Op::AddPlain) {raw=abs(ma)*a.coeff+abs(mb*c);error=abs(ma)*a.error+*enc.bias;add(Role::PlainAdd,0,ma,mb);}
      else {raw=abs(c)*a.coeff;error=abs(c)*a.error+*enc.bias;add(Role::PlainMul,0,c);}
    } else return;
    break;
  }
  default:return;
  }
  // Before modular centering, inverse canonical embedding bounds each integer
  // coefficient by the maximum of ALL conjugate slot magnitudes. Never apply
  // this lemma to just selected lanes or to an already reduced candidate.
  error+=*drift;
  const Exact slot=I+error, candidate=floor(std::min(raw,slot));
  put("coefficient_route",raw);put("embedding_route",slot);put("coeff",candidate);
  put("error",error);put("degree",degree);
  // Standard residue centering and v6.2 PolyToBigintCentered disagree at +floor(Q/2).
  // The stronger active-Q guard is sufficient for both conventions.
  const bool fits=2*candidate<Q-1;
  if(fits) out.arithmetic=ArithmeticSupport{degree,candidate,I,error};
  for(const auto &x:effects) {
    auto &r=row(x.role);r.premises.clear();r.centered_support=facts;r.candidate_coeff_abs=candidate;
    r.dc_before_center.reset();r.dc_after_center.reset();r.dc_center_correction.reset();
    r.lhs_gain=x.lhs;r.rhs_gain=x.rhs;r.formula.clear();r.bias.reset();
    if(!fits) {r.blocker="unproved_common_centered_lift";continue;}
    r.formula=std::string("b2k/centered-")+name(x.role)+"-v1";r.blocker.clear();r.bias=x.bias;
  }
}
} // namespace snabf::ir::noise

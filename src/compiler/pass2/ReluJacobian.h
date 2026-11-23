#pragma once

#include "ReluBernstein.h"
#include <mpfi.h>
#include <map>
#include <set>
#include <limits>
#include <memory>
#include <optional>
#include <string>

// A scalar holomorphic circuit. Each uncertain literal is one labelled complex
// disk, not a new draw on each use. Anchor restarts a short exact polynomial in
// the previous stage's REAL nominal value; it never restarts the perturbations.
namespace snabf::pass2::relu_jacobian {
using Q=mpq_class;
using Poly=std::vector<Q>;
using Id=size_t;
enum class Op { Input, Literal, Add, Mul, Anchor };
struct Node { Op op; Id a=0,b=0; Q value=0, radius=0; std::string name; std::optional<Q> fixedDelta; };
struct Circuit {
  std::vector<Node> nodes;
  Id push(Node n) { nodes.push_back(std::move(n)); return nodes.size()-1; }
  Id input(Q e,std::string name) {return push({Op::Input,0,0,0,e,std::move(name)});}
  Id literal(Q v,Q e=0,std::string name={}) {return push({Op::Literal,0,0,v,e,std::move(name)});}
  Id add(Id a,Id b){return push({Op::Add,a,b});}
  Id mul(Id a,Id b){return push({Op::Mul,a,b});}
  Id anchor(Id a){return push({Op::Anchor,a});}
};
struct Interval { double lo=0,hi=0; };
struct Disk { Interval real; double r=0; }; // real interval + zero-centred COMPLEX disk
using Grad=std::map<Id,Disk>;
struct Bound { Disk value; Grad grad; };
struct Cover { std::vector<double> magnitude,error; std::map<Id,double> outputGain; unsigned cells=0; };
inline void need(bool ok,const char *s){if(!ok)throw std::invalid_argument(s);}
inline double finite(double x){if(!std::isfinite(x))throw std::overflow_error("unbounded ReLU Jacobian enclosure");return x;}
inline double plus(double x,double y){if(x==0||y==0)return x==0?y:x;return finite(std::nextafter(x+y,INFINITY));}
inline double times(double x,double y){if(x==0||y==0)return 0;if(x==1||y==1)return x==1?y:x;return finite(std::nextafter(x*y,INFINITY));}
inline double mag(Interval x){return std::max(std::abs(x.lo),std::abs(x.hi));}
inline double mag(Disk x){return plus(mag(x.real),x.r);}
struct MP {
  mpfi_t x;
  MP(){mpfi_init2(x,160);} explicit MP(Interval v):MP(){mpfi_interv_d(x,v.lo,v.hi);}
  MP(const MP&)=delete; ~MP(){mpfi_clear(x);}
  Interval get()const {
    mpfr_t t;mpfr_init2(t,160);mpfi_get_left(t,x);double l=mpfr_get_d(t,MPFR_RNDD);
    mpfi_get_right(t,x);double h=mpfr_get_d(t,MPFR_RNDU);mpfr_clear(t);return {finite(l),finite(h)};
  }
};
inline Interval enclose(const Q &lo,const Q &hi){MP r;mpfi_interv_q(r.x,lo.get_mpq_t(),hi.get_mpq_t());return r.get();}
inline Interval add(Interval a,Interval b){MP x(a),y(b),r;mpfi_add(r.x,x.x,y.x);return r.get();}
inline Interval mul(Interval a,Interval b){MP x(a),y(b),r;mpfi_mul(r.x,x.x,y.x);return r.get();}
struct PointWitness { Interval diff; std::vector<int> signs; };
// Pick ONE admissible real vector in the simultaneous complex box. Floating
// gradients only choose signs. BOTH compared values are separately evaluated
// by MPFI from exact rationals, using the actual signed scale offsets. This is
// a lower witness for the engineering model, never a decryption/upper proof.
inline PointWitness pointWitness(const Circuit &c,const Q &x,int direction) {
  need(!c.nodes.empty()&&(direction==1||direction==-1),"invalid witness request");
  std::vector<double> v,adj(c.nodes.size());PointWitness w;w.signs.resize(c.nodes.size());
  for(const auto &n:c.nodes){double a=0;
    switch(n.op){case Op::Input:a=x.get_d();break;case Op::Literal:a=n.value.get_d();break;
    case Op::Anchor:a=v.at(n.a);break;case Op::Add:a=v.at(n.a)+v.at(n.b);break;case Op::Mul:a=v.at(n.a)*v.at(n.b);break;}v.push_back(a);}
  adj.back()=1;for(size_t i=c.nodes.size();i--;){auto &n=c.nodes[i];double a=adj[i];
    if(n.op==Op::Anchor||n.op==Op::Add)adj[n.a]+=a;
    if(n.op==Op::Add)adj[n.b]+=a;
    if(n.op==Op::Mul){adj[n.a]+=a*v[n.b];adj[n.b]+=a*v[n.a];}
    w.signs[i]=direction*((a>0)-(a<0));}
  auto evaluate=[&](bool perturbed){std::vector<std::unique_ptr<MP>> vals;
    for(size_t i=0;i<c.nodes.size();++i){auto &n=c.nodes[i];auto y=std::make_unique<MP>();
      switch(n.op){case Op::Input:case Op::Literal:{Q k=n.op==Op::Input?x:n.value;
        if(perturbed){Q offset=n.fixedDelta.value_or(Q(w.signs[i]*n.radius));need(abs(offset)<=n.radius,"unbound model offset");k+=offset;}
        mpfi_set_q(y->x,k.get_mpq_t());break;}
      case Op::Add:mpfi_add(y->x,vals.at(n.a)->x,vals.at(n.b)->x);break;
      case Op::Mul:mpfi_mul(y->x,vals.at(n.a)->x,vals.at(n.b)->x);break;
      case Op::Anchor:mpfi_set(y->x,vals.at(n.a)->x);break;}
      vals.push_back(std::move(y));}return std::move(vals.back());};
  auto y=evaluate(true),y0=evaluate(false);MP diff;mpfi_sub(diff.x,y->x,y0->x);w.diff=diff.get();return w;
}
inline double tail(Disk a,Disk b){return plus(plus(times(mag(a.real),b.r),times(mag(b.real),a.r)),times(a.r,b.r));}
inline Disk add(Disk a,Disk b){return {add(a.real,b.real),plus(a.r,b.r)};}
inline Disk mul(Disk a,Disk b){return {mul(a.real,b.real),tail(a,b)};}
inline void trim(Poly &p){while(!p.empty() && p.back()==0)p.pop_back();}
// p(Z)+rem for the SAME complex stage input Z=u+delta. The Taylor jet
// accounts for all powers of delta with their signed polynomial coefficients.
// rem contains only local injection effects, including all mixed products.
struct Symbol { Poly p; Disk d; double rem,zmag; };
struct Segment {
  Interval u; double delta; std::map<Poly,Interval> cache;std::map<Poly,double> jets;
  Interval range(const Poly &p){
    auto it=cache.find(p);
    if(it==cache.end()) {
      Interval v{};
      if(p.size()==1)v=enclose(p[0],p[0]);
      if(p.size()>1){auto a=relu_bernstein::stage(p,{Q(u.lo),Q(u.hi)});v=enclose(a.lo,a.hi);}
      it=cache.emplace(p,v).first;
    }
    return it->second;
  }
  Symbol symbol(Poly p,double r=0){
    trim(p);need(p.size()<=16,"missing stage boundary / excessive exact degree");
    auto center=range(p);auto it=jets.find(p);
    if(it==jets.end()){
      double tail=0,power=1; Poly derivative=p;
      for(size_t k=1;k<p.size() && delta!=0;++k){
        for(size_t j=0;j+1<derivative.size();++j)derivative[j]=derivative[j+1]*(j+1)/k;
        derivative.pop_back();power=times(power,delta);tail=plus(tail,times(power,mag(range(derivative))));
      }it=jets.emplace(p,tail).first;
    }
    return {std::move(p),{center,plus(it->second,r)},r,plus(mag(center),it->second)};
  }
  Symbol sum(const Symbol &a,const Symbol &b){
    Poly p=a.p;p.resize(std::max(p.size(),b.p.size()));
    for(Id i=0;i<b.p.size();++i)p[i]+=b.p[i];
    return symbol(std::move(p),plus(a.rem,b.rem));
  }
  Symbol product(const Symbol &a,const Symbol &b){
    Poly p;if(!a.p.empty()&&!b.p.empty())p.resize(a.p.size()+b.p.size()-1);
    for(Id i=0;i<a.p.size();++i)for(Id j=0;j<b.p.size();++j)p[i+j]+=a.p[i]*b.p[j];
    return symbol(std::move(p),plus(plus(times(a.zmag,b.rem),times(b.zmag,a.rem)),times(a.rem,b.rem)));
  }
};
inline constexpr Id anchorKey=std::numeric_limits<Id>::max();
struct Local { Id segment=anchorKey; Symbol value; std::map<Id,Symbol> grad; };
inline constexpr Disk one{{1,1},0};

// All domain and perturbation boxes are CLOSED. The straight-line path t*eta
// stays in their product for 0<=t<=1. Integrating the full-box Jacobian gives
// sum_j sup|dF/deta_j| rho_j; product-rule Jacobians use perturbed operands,
// so mixed terms are covered even when all nominal partials vanish.
inline std::vector<Bound> cell(const Circuit &c,Interval x) {
  need(!c.nodes.empty()&&c.nodes[0].op==Op::Input&&std::isfinite(x.lo)&&std::isfinite(x.hi)&&x.lo<=x.hi,"invalid Jacobian input");
  std::vector<Bound> out;std::vector<Local> locals;std::vector<double> radii;
  Id tag=0,base=0;Segment seg{x,0,{},{}};std::set<std::string> labels;
  for(Id i=0;i<c.nodes.size();++i){
    const auto &n=c.nodes[i];Bound y;Local l;l.segment=tag;
    need(n.radius>=0,"negative perturbation radius");
    need(!n.fixedDelta || (n.op==Op::Literal && abs(*n.fixedDelta)<=n.radius),"unbound deterministic offset");
    if(n.radius!=0)need(!n.name.empty()&&labels.insert(n.name).second,"missing/duplicate perturbation id");
    if(n.op!=Op::Input&&n.op!=Op::Literal)need(n.radius==0,"unbound circuit perturbation");
    radii.push_back(enclose(n.radius,n.radius).hi);
    if(n.op==Op::Input||n.op==Op::Anchor){
      need(n.op==Op::Input?i==0:n.a<i,"bad anchor");
      if(i)y=out.at(n.a);else {y.value={x,radii[i]};y.grad[i]=one;}
      base=i;tag=i;l.segment=tag;seg={y.value.real,y.value.r,{},{}};
      l.value=seg.symbol({0,1});l.grad[anchorKey]=seg.symbol({1});
    } else if(n.op==Op::Literal){
      l.value=seg.symbol({n.value},radii[i]);y.value=l.value.d;
      if(radii[i]){l.grad[i]=seg.symbol({1});y.grad[i]=one;}
    } else {
      need(n.a<i&&n.b<i&&(n.op==Op::Add||n.op==Op::Mul),"invalid primitive or SSA reference");
      const bool prod=n.op==Op::Mul;const auto &a=out[n.a],&b=out[n.b];
      const auto &la=locals[n.a],&lb=locals[n.b];
      if(la.segment==tag&&lb.segment==tag){
        l.value=prod?seg.product(la.value,lb.value):seg.sum(la.value,lb.value);
        auto acc=[&](const Local &s,const Symbol &other){for(const auto &[j,v]:s.grad){
          auto t=prod?seg.product(v,other):v;
          auto [it,inserted]=l.grad.emplace(j,t);if(!inserted)it->second=seg.sum(it->second,t);
        }};
        acc(la,lb.value);acc(lb,la.value);y.value=l.value.d;
        for(const auto &[j,v]:l.grad){
          if(j==anchorKey){for(const auto &[k,g]:out[base].grad)y.grad.emplace(k,mul(g,v.d));}
          else y.grad.emplace(j,v.d);
        }
      } else {
        // The explicit original-x bypass crosses a segment boundary. Accumulate
        // both paths under the SAME global coordinate before taking magnitudes.
        l.segment=anchorKey;y.value=prod?mul(a.value,b.value):add(a.value,b.value);
        auto acc=[&](const Bound &s,Disk other){for(const auto &[j,v]:s.grad){
          auto t=prod?mul(v,other):v;
          auto [it,inserted]=y.grad.emplace(j,t);if(!inserted)it->second=add(it->second,t);
        }};
        acc(a,b.value);acc(b,a.value);
      }
      double r=0;for(const auto &[j,v]:y.grad)r=plus(r,times(mag(v),radii.at(j)));
      y.value.r=std::min(y.value.r,r);l.value.d.r=y.value.r;
    }
    out.push_back(std::move(y));locals.push_back(std::move(l));
  }
  return out;
}
inline Cover prove(const Circuit &c,const Q &lo,const Q &hi,unsigned subdivisions,unsigned octaves=0,
                   unsigned refine=0,double target=0) {
  need(lo<=0&&hi>=0&&subdivisions&&subdivisions<=65536&&octaves<=32,"invalid closed ReLU cover");
  need(refine<=12 && (!refine || (std::isfinite(target)&&target>0)),"invalid ReLU refinement budget");
  Cover p;p.magnitude.resize(c.nodes.size());p.error.resize(c.nodes.size());
  auto visit=[&](auto &self,const Q &a,const Q &b,unsigned depth)->void{
    std::vector<Bound> v;
    try{v=cell(c,enclose(a,b));}
    catch(const std::overflow_error&){if(!depth)throw std::overflow_error("unbounded ReLU Jacobian on ["+a.get_str()+","+b.get_str()+"]");}
    if(depth && (v.empty() || v.back().value.r>target)){
      Q mid=(a+b)/2;self(self,a,mid,depth-1);self(self,mid,b,depth-1);return;
    }
    for(Id i=0;i<v.size();++i){p.magnitude[i]=std::max(p.magnitude[i],mag(v[i].value));p.error[i]=std::max(p.error[i],v[i].value.r);}
    for(auto &[k,g]:v.back().grad)p.outputGain[k]=std::max(p.outputGain[k],mag(g));++p.cells;
  };
  for(const Q &end:{lo,hi}){Q prev=0;
    for(int k=octaves;k>=0;--k){Q next=end/(mpz_class(1)<<k);
    Q a=std::min(prev,next),b=std::max(prev,next);prev=next;
    for(unsigned j=0;j<subdivisions;++j){
      visit(visit,a+(b-a)*j/subdivisions,a+(b-a)*(j+1)/subdivisions,refine);
    }}
  }return p;
}
} // namespace snabf::pass2::relu_jacobian

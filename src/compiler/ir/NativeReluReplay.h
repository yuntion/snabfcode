#pragma once
#include "NativePhysical.h"
#include "pass2/ReluJacobian.h"

namespace snabf::native {
// All circuit values are raw slot values divided by their own exact metadata
// scale. The nominal point is the source DAG (signed binary64 coefficients),
// not a rounded encoder invocation. A shared wire keeps its coordinate ID.
struct NativeReluReplay {
  using R=ExactRational;
  using Id=pass2::relu_jacobian::Id;
  pass2::relu_jacobian::Circuit circuit;
  const NativeArithmeticProgram::ReluSpan &span;
  std::map<std::string,Id> ids;
  std::vector<Id> edges;
  struct JoinStep { std::string name; Id value; R scale; int level; };
  std::vector<JoinStep> partials;
  std::vector<R> targets;
  // Propose scale targets using the same log-linear equations as the DAG
  // scheduler, without binary64 cancellation in differences of log scales.
  // This is not proof arithmetic: encode(), rescale and every actual join
  // below are still verified on the resulting exact metadata and payload.
  static R trans(const R &arg,bool exponential) {
    mpfr_t t;mpfr_init2(t,256);mpfr_set_q(t,arg.get_mpq_t(),MPFR_RNDN);
    if(exponential) mpfr_exp2(t,t,MPFR_RNDN);else mpfr_log2(t,t,MPFR_RNDN);
    if(!mpfr_number_p(t) || (!mpfr_zero_p(t) && std::abs(mpfr_get_exp(t))>16384)){
      mpfr_clear(t);throw std::invalid_argument("unsupported ReLU scale target");}
    R out;mpfr_get_q(out.get_mpq_t(),t);mpfr_clear(t);return out;
  }
  static std::vector<R> scaleTargets(const pass3::Poly &p,const pass3::Solution &s) {
    const size_t n=p.nodes.size();std::vector<std::vector<R>> rows;std::vector<size_t> pivots;
    auto row=[&](size_t i)->auto&{auto &r=rows.emplace_back(n+1);r.at(i)=1;return r;};
    const R base=R(mpz_class(1)<<s.scale);row(0);row(p.output);
    for(size_t i=1;i<n;++i){const auto &node=p.nodes[i];
      if(node.op=="mul"){auto &r=row(i);for(int a:node.args)--r.at(a);
        r[n]=trans(base/R(s.q.at(p.level-node.depth+1)),false);}
      else if(node.op=="add"){for(int a:node.args)--row(i).at(a);}
    }
    size_t k=0;
    for(size_t i=0;i<n&&k<rows.size();++i){size_t pivot=k;while(pivot<rows.size()&&!rows[pivot][i])++pivot;
      if(pivot==rows.size())continue;std::swap(rows[k],rows[pivot]);R f=rows[k][i];
      for(size_t j=i;j<=n;++j)rows[k][j]/=f;
      for(size_t h=0;h<rows.size();++h){if(h==k || !rows[h][i])continue;R g=rows[h][i];
        for(size_t j=i;j<=n;++j)rows[h][j]-=g*rows[k][j];}
      ++k;pivots.push_back(i);
    }
    std::vector<R> v(n,base);for(size_t i=0;i<k;++i)v[pivots[i]]=trans(R(s.scale)+rows[i][n],true);
    return v;
  }
  NativeReluReplay(const NativeArithmeticProgram::ReluSpan &s,const std::string &entry,const R &err):span(s) {
    ids[entry]=circuit.input(err,"entry:"+entry);
  }
  Id factor(Id v,const R &actual,const std::string &name) {
    if(actual==1)return v;
    auto k=circuit.literal(1,abs(actual-1),name+":scale");circuit.nodes[k].fixedDelta=actual-1;
    return circuit.mul(v,k);
  }
  Id fresh(Id v,const R &radius,const std::string &name) {
    if(!radius)return v;
    return circuit.add(v,circuit.literal(0,radius,name));
  }
  Id encoded(const R &value,const R &gain,const R &noise,const std::string &name) {
    auto v=circuit.literal(value);v=factor(v,gain,name);
    return fresh(v,noise,name+":encoding");
  }
  void join(const std::string &id,const std::string &a,const R &lhs,const std::string &b,const R &rhs,
            const R &m,int level) {
    auto ax=factor(ids.at(a),lhs,id+":lhs");auto bx=factor(ids.at(b),rhs,id+":rhs");
    auto value=circuit.add(ax,bx); ids[id]=value;
    partials.push_back({id+":lhs",ax,m,level});partials.push_back({id+":rhs",bx,m,level});
    partials.push_back({id,value,m,level});
  }
  void finishEdge(int i,const std::string &id) {
    auto v=ids.at(id);edges.push_back(v);
    if(span.stageEnds.contains(i))ids[id]=circuit.anchor(v);
  }
};
} // namespace snabf::native

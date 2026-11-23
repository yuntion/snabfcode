#pragma once

#include "PolyTemplates.h"
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace snabf::pass2::relu_bernstein {
using Q = mpq_class;
struct Axis { Q lo, hi; };
using Row = std::vector<Q>;
inline unsigned choose(unsigned n,unsigned k) {
  unsigned c=1; for(unsigned i=1;i<=k;++i) c=c*(n+1-i)/i; return c;
}
inline Axis hull(const Row &b,unsigned depth,unsigned &work) {
  const auto [l,h]=std::minmax_element(b.begin(),b.end());
  const auto [a,z]=std::minmax(b.front(),b.back());
  if ((*l==a && *h==z) || depth==20 || !work) return {*l,*h};
  --work;
  Row tmp=b, left,right;
  for(size_t n=tmp.size();n>0;--n) {
    left.push_back(tmp.front()); right.push_back(tmp[n-1]);
    for(size_t i=0;i+1<n;++i) tmp[i]=(tmp[i]+tmp[i+1])/2;
  }
  std::reverse(right.begin(),right.end());
  auto x=hull(left,depth+1,work), y=hull(right,depth+1,work);
  return {std::min(x.lo,y.lo),std::max(x.hi,y.hi)};
}
inline Q snap(const Q &q,bool upper) {
  const mpz_class unit=mpz_class(1)<<160, scaled=q.get_num()*unit;
  mpz_class n;
  if(upper) mpz_cdiv_q(n.get_mpz_t(),scaled.get_mpz_t(),q.get_den_mpz_t());
  else mpz_fdiv_q(n.get_mpz_t(),scaled.get_mpz_t(),q.get_den_mpz_t());
  Q out(n,unit); out.canonicalize(); return out;
}
inline Axis stage(const Row &p,const Axis &x) {
  const unsigned d=p.size()-1; Row lo(d+1,1),span(d+1,1),shift(d+1),b(d+1);
  for(unsigned i=1;i<=d;++i) {lo[i]=lo[i-1]*x.lo; span[i]=span[i-1]*(x.hi-x.lo);}
  for(unsigned j=0;j<=d;++j) {
    for(unsigned i=j;i<=d;++i) shift[j]+=p[i]*choose(i,j)*lo[i-j];
    shift[j]*=span[j];
    for(unsigned k=j;k<=d;++k) b[k]+=shift[j]*choose(k,j)/choose(d,j);
  }
  unsigned work=512; auto y=hull(b,0,work); return {snap(y.lo,false),snap(y.hi,true)};
}
struct Proof { std::array<Q,2> upper; std::array<unsigned,2> cells{}; };
inline Proof prove(const std::vector<StagePolynomial> &chain,Interval1D input,double budget) {
  if(chain.empty() || chain.size()>8 || !std::isfinite(input.lo) || !std::isfinite(input.hi) ||
     !(input.lo<=0 && input.hi>=0 && std::isfinite(budget) && budget>0))
    throw std::invalid_argument("invalid rational ReLU domain or budget");
  std::vector<Row> rows;
  for(const auto &p:chain) {
    if(p.degree>15 || p.coefficients.size()!=p.degree+1) throw std::invalid_argument("invalid rational ReLU degree");
    auto &row=rows.emplace_back();
    for(size_t i=0;i<p.coefficients.size();++i) {
      double a=p.coefficients[i]; if(!std::isfinite(a) || (!(i%2) && a!=0)) throw std::invalid_argument("non-odd rational ReLU stage");
      row.emplace_back(a);
    }
  }
  struct Cell { Axis x; unsigned depth; };
  Proof proof;
  for(unsigned side=0;side<2;++side) {
    std::vector<Cell> todo{{side ? Axis{0,Q(input.hi)} : Axis{Q(input.lo),0},0}};
    while(!todo.empty()) {
      Cell cell=todo.back(); todo.pop_back(); Axis z=cell.x;
      for(const auto &p:rows) z=stage(p,z);
      Q e=0;
      for(const Q &x:{cell.x.lo,cell.x.hi}) for(const Q &y:{z.lo,z.hi}) {
        Q r=abs(x*(y+(side?-1:1))/2); if(r>e)e=r;
      }
      if(e<=Q(budget)) {++proof.cells[side]; if(e>proof.upper[side])proof.upper[side]=e; continue;}
      if(cell.depth>=32 || todo.size()+proof.cells[side]>=2048)
        throw std::invalid_argument("rational Bernstein cover did not establish ReLU budget");
      const Q m=(cell.x.lo+cell.x.hi)/2;
      todo.push_back({{m,cell.x.hi},cell.depth+1}); todo.push_back({{cell.x.lo,m},cell.depth+1});
    }
  }
  return proof;
}
} // namespace snabf::pass2::relu_bernstein

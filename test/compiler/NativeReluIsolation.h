#pragma once
#include "ir/NativeArithmetic.h"
#include "pass3/ConcreteVerifier.h"
#include "pass3/PrimeInstantiator.h"
#include "llvm/ADT/APFloat.h"
#include <cmath>
#include <gmpxx.h>
#include <set>

namespace snabf::native::test {
// A lower bound on the *verifier's allowance*, not on error in a ciphertext.
// Use only the source rounding term at the highest permitted scale. Discard
// all subsequent local terms and the quadratic term. Nonnegative gains and
// rounding each multiply/add downward preserve the inequality by induction.
inline llvm::json::Object sourceOnlyFloor(const pass3::Graph &g,const pass3::Options &o) {
  const double limit=8*o.tolerance, root=std::ldexp(pass3::rounding(g.n),-o.maxScale);
  llvm::json::Object row{{"max_scale",o.maxScale},{"source_floor",root},{"limit",limit}};
  std::vector<double> floors;
  for(const auto &e:g.edges) {
    if(e.ring && e.ring!=g.n) throw std::invalid_argument("single-ring diagnostic required");
    llvm::APFloat sum(e.op==pass3::Op::Source?root:0.);
    for(size_t j=0;j<e.inputs.size();++j) {
      double k=e.gains.at(j);
      if(!(std::isfinite(k) && k>=0))throw std::invalid_argument("nonnegative finite gain required");
      llvm::APFloat term(floors.at(e.inputs[j]));term.multiply(llvm::APFloat(k),llvm::APFloat::rmTowardNegative);
      sum.add(term,llvm::APFloat::rmTowardNegative);
    }
    double lower=sum.convertToDouble();
    if(lower>limit) {row["first_edge"]=e.id;row["allowance_floor"]=lower;break;}
    floors.push_back(lower);
  }
  return row;
}

inline llvm::json::Object isolate(const NativeArithmeticProgram &program,const pass3::Options &opts) {
  using llvm::json::Array;
  using llvm::json::Object;
  const auto &g=program.graph;
  if(g.n!=65536 || g.levels!=29 || opts.alpha!=2 || program.reluSlotDomains.empty())
    throw std::invalid_argument("expected concrete native ReLU profile");
  struct Arm {const char *name;int scale,base,lowCount,lowBits,pBits;};
  Array arms;
  for(auto a:{Arm{"fixed43",43,53,0,43,54},Arm{"uniform44",44,51,0,44,55},
              Arm{"tail45",45,50,6,40,56},Arm{"tail46",46,50,8,40,54}}) {
    pass3::Solution s; std::set<int64_t> used;
    s.scale=a.scale;s.k.assign(g.edges.size(),0);
    for(int i=0;i<g.levels;++i) s.q.push_back(pass3::nearestPrime(g.n,i==0?a.base:i<=a.lowCount?a.lowBits:a.scale,used));
    for(int i=0;i<2;++i)s.p.push_back(pass3::nearestPrime(g.n,a.pBits,used));
    std::string why;
    try {why=pass3::replay(g,opts,s).reason;} catch(const std::runtime_error &e){why=e.what();}
    Array q,p; mpz_class qp=1;
    for(auto v:s.q){q.push_back(v);qp*=v;} for(auto v:s.p){p.push_back(v);qp*=v;}
    arms.push_back(Object{{"arm",a.name},{"ring",g.n},{"q",std::move(q)},{"p",std::move(p)},
      {"qp_bits",int64_t(mpz_sizeinbase(qp.get_mpz_t(),2))},{"scale",a.scale},{"security_cap",opts.securityCap},
      {"accepted",why.empty()},{"completed_edges",int64_t(s.errors.size())},{"reason",why}});
  }
  return Object{{"schema","snabf.native-relu-isolation.v1"},{"source",program.source.inspection.getString("source_sha256")->str()},
    {"input_spec",program.source.inspection.getString("input_spec_sha256")->str()},
    {"arms",std::move(arms)},{"source_only_floor",sourceOnlyFloor(g,opts)}};
}
} // namespace snabf::native::test

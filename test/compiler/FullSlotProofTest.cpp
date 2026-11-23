#include "pass4/FullSlotProof.h"
#include <iostream>

using namespace snabf;
using namespace snabf::pass4;
void check(bool ok,const char *msg){if(!ok)throw std::runtime_error(msg);}
int main(){
  constexpr int S=4096;
  FullSlotWeights w{S+1,S+1,{}};
  int j=0;
  for(int k:{0,1,S,S+1,2*S-1}){
    auto &v=w.diagonals[k];v.resize(2*S);
    for(int i=0;i<w.outputWidth;++i) if((i+k)%(2*S)<w.inputWidth)
      v[i]=std::array<double,5>{.25,-.5,.125,.375,-.0625}[j]*(1+(i+3*j)%5)/4;
    ++j;
  }
  auto a=analyzeFullSlot(w,S);
  w.diagonals.at(0).back()=.125;
  try{analyzeFullSlot(w,S);return 1;}catch(const std::invalid_argument &){}
  check(a.sourceTerms==5 && a.maxChildTerms>0 && a.norm>0,"bad support");
  pass3::Solution old; old.q={1125899908022273,1099511922689,1099512938497,1099510054913,1099514314753,1099514478593};
  old.p={1125899908612097};old.scale=40;
  auto oldDown=fullSlotMargin(a,old,true),oldKeep=fullSlotMargin(a,old,false);
  ExactRational eps(1,4096);
  check(oldDown.downKeyCharge>eps && oldKeep.downKeyCharge==0 && oldDown.total>eps,"missing fresh conversion noise");
  auto f=solveFullSlot(a,352);
  check(f.oldDown.total>=eps && f.initial.q!=f.basis.q && f.basis.scale>f.initial.scale,
        "expected an actual source-basis repair, not a repeated identical solve");
  for(auto &m:{f.keep,f.down})check(m.headroom && m.gain>0 && m.total==m.fresh+m.gain*m.cutError &&
    m.budget==(eps-m.fresh)/m.gain && m.cutError<m.budget,"bad physical cut transfer");
  check(f.stats.solverRuns && f.stats.modelsWithCuts && pass2::exactBinary(f.submittedBudget)<=std::min(f.oldKeep.budget,f.oldDown.budget),"cut not submitted");
  std::cout<<"d="<<f.basis.scale<<" models="<<f.stats.modelsWithCuts<<" runs="<<f.stats.solverRuns<<" seedDown="<<f.oldDown.total.get_d()<<" keep="<<f.keep.total.get_d()<<" down="<<f.down.total.get_d()<<" row="<<f.submittedBudget<<'\n';
}

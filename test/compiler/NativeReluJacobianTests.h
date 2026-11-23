#pragma once
#include "pass2/ReluJacobian.h"
#include <iostream>
inline void nativeReluJacobianTests(){using namespace snabf::pass2::relu_jacobian;
  {Circuit c;auto x=c.input(Q(1,16),"x");auto a=c.anchor(x);
   c.add(x,c.mul(a,c.literal(-1)));auto p=prove(c,-1,1,4);
   need(p.cells==8&&p.error.back()==0&&p.outputGain.at(x)==0,"shared x bypass was not cancelled");}
  {Circuit c;c.input(0,"x");auto a=c.literal(0,Q(1,4),"a"),b=c.literal(0,Q(1,8),"b");c.mul(a,b);
   auto p=prove(c,0,0,1);need(p.error.back()>=1./32&&p.error.back()<0.0312500001,"missed simultaneous complex mixed term");
   need(p.outputGain.at(a)>=.125&&p.outputGain.at(b)>=.25,"partials evaluated only at nominal point");}
  {Circuit c;auto x=c.input(Q(1,32),"x");auto x2=c.mul(x,x);auto a=c.anchor(x2);
   auto k=c.literal(Q(-3,2),Q(1,64),"coef");auto y=c.mul(k,a);c.add(y,x);
   auto p=prove(c,-1,1,8);need(p.outputGain.at(x)>=4 && p.outputGain.at(k)>1,"signed chain rule on thickened domain");
   need(p.error.back()>=4./32+1./64 && p.error.back()<.15,"cross-anchor jointly bounded increments");}
  {Circuit c;c.input(0,"x");c.literal(0,1,"same");c.literal(1,1,"same");bool refused=false;
   try{prove(c,-1,1,1);}catch(const std::invalid_argument&){refused=true;}need(refused,"duplicate perturbation id");}
  {Circuit c;auto x=c.input(Q(1,16),"x");auto k=c.literal(2,Q(1,8),"encoded");c.nodes[k].fixedDelta=Q(-1,8);c.mul(x,k);
   auto w=pointWitness(c,1,1);Q expected=Q(17,16)*Q(15,8)-2;
   need(Q(w.diff.lo)<=expected && Q(w.diff.hi)>=expected,"signed physical offset lost in point replay");
   auto p=prove(c,-1,1,2,2,2,.1);need(p.error.back()>=.25 && p.cells>12,"incomplete adaptive cover");}
  std::cout<<"Shared cancellation, simultaneous mixed disk, weighted bypass, signed actual offsets and adaptive Jacobian gates passed\n";
}

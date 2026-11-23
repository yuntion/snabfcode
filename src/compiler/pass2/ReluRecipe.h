#pragma once

#include "PolyTemplates.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace snabf::pass2 {
// A fixed compact search arm. The offline minimax seeds have NO authority:
// the 160-bit enclosure is recomputed from the emitted binary64 powers.
inline std::vector<StagePolynomial> reluRecipe(Interval1D input) {
  const double r=std::max(-input.lo,input.hi);
  if (!(std::isfinite(input.lo) && std::isfinite(input.hi) && input.lo<=0 && input.hi>=0))
    throw std::invalid_argument("ReLU needs a finite closed domain containing zero");
  // Normalize by an enclosing power of two, never by an observed slot range.
  int exp=0;
  if (r>0 && std::frexp(r,&exp)==0.5) --exp;
  std::vector<StagePolynomial> result;
  auto add=[&](std::initializer_list<double> odd) {
    const unsigned d=2*static_cast<unsigned>(odd.size())-1;
    StagePolynomial p{d,std::vector<double>(d+1)};
    unsigned k=1;
    for (auto c:odd) {
      p.coefficients[k]=std::scalbn(c,result.empty() ? -exp*int(k) : 0);
      if (!std::isfinite(p.coefficients[k]) || !p.coefficients[k])
        throw std::overflow_error("unrepresentable normalized ReLU coefficient");
      k+=2;
    }
    result.push_back(std::move(p));
  };
  add({0x1.7b3491a524d7bp+2,-0x1.182735c9c1134p+5,0x1.03790354cb3ecp+6,-0x1.1e297fe47405cp+5});
  add({0x1.7990747cb27cap+2,-0x1.14c6e37fce021p+5,0x1.ff86a949e4e72p+5,-0x1.19c34d4e83197p+5});
  add({0x1.6fd2a8dbad7c0p+2,-0x1.014bca98bad70p+5,0x1.d4e419b0f4c11p+5,-0x1.00880232c50dep+5});
  add({0x1.39d636ac9208fp+2,-0x1.4b7ef983858dep+4,0x1.11c75a6be7a1ep+5,-0x1.1d04625b619dep+4});
  add({0x1.6bb164db302b6p+1,-0x1.299becd306a9fp+2,0x1.17754359a6e84p+2,-0x1.8ffc9a05498e2p+0});
  add({0x1.e12229fdc167cp+0,-0x1.42453d8b0d622p+0,0x1.848c4e352fe9bp-2});
  return result;
}
}  // namespace snabf::pass2

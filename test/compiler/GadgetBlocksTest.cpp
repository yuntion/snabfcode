#include "pass3/GadgetNoiseContract.h"
#include <set>
#include <stdexcept>
using snabf::pass3::activeDigitBlocks;
int main() {
  for (int ell = 0; ell <= 31; ++ell)
    for (int alpha : {1, 2, 3, 4})
      for (int k = 0; k <= ell; ++k) {
        std::set<int> survivingKeys;
        for (int j = k; j <= ell; ++j) survivingKeys.insert(j / alpha);
        if (activeDigitBlocks(ell, k, alpha) != int(survivingKeys.size())) return 1;
      }
  if (activeDigitBlocks(2, 1, 2) != 2 || activeDigitBlocks(2, 2, 2) != 1) return 2;
  for (int k : {-1, 3}) {
    try { activeDigitBlocks(2, k, 2); return 3; }
    catch (const std::invalid_argument &) {}
  }
  return 0;
}

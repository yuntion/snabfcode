#include "PrimeInstantiator.h"
#include <cmath>
#include <stdexcept>
namespace snabf::pass3 {
namespace {
uint64_t mul(uint64_t a, uint64_t b, uint64_t m) {
  return (__uint128_t)a * b % m;
}
uint64_t power(uint64_t a, uint64_t b, uint64_t m) {
  uint64_t r = 1;
  for (; b; b >>= 1, a = mul(a, a, m))
    if (b & 1)
      r = mul(r, a, m);
  return r;
}
} // namespace
bool prime(uint64_t n) {
  if (n < 2)
    return false;
  for (auto p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    if (n % p == 0)
      return n == uint64_t(p);
  }
  uint64_t d = n - 1;
  int s = 0;
  while (!(d & 1)) {
    d >>= 1;
    ++s;
  }
  for (uint64_t a : {2, 325, 9375, 28178, 450775, 9780504, 1795265022}) {
    if (!(a % n))
      continue;
    auto x = power(a % n, d, n);
    if (x == 1 || x == n - 1)
      continue;
    int j = 1;
    for (; j < s; ++j) {
      x = mul(x, x, n);
      if (x == n - 1)
        break;
    }
    if (j == s)
      return false;
  }
  return true;
}
int64_t nearestPrime(int n, double bits, std::set<int64_t> &used) {
  if (n < 4 || (n & (n - 1)) || !(bits >= 18 && bits <= 60))
    throw std::invalid_argument("invalid NTT width/ring");
  long double target = std::exp2((long double)bits);
  int64_t step = 2LL * n, lo = (int64_t(target) - 1) / step * step + 1,
          hi = lo + step;
  const int64_t cap = 1LL << 60;
  for (int i = 0; i < 2000000; ++i) {
    int64_t v;
    if (lo > 2 && (hi >= cap || target - lo <= hi - target)) {
      v = lo;
      lo -= step;
    } else {
      v = hi;
      hi += step;
    }
    if (v > 2 && v < cap && !used.contains(v) && prime(v)) {
      used.insert(v);
      return v;
    }
  }
  throw std::runtime_error("no admissible NTT prime");
}
} // namespace snabf::pass3

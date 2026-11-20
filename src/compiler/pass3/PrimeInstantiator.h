#pragma once
#include <cstdint>
#include <set>
namespace snabf::pass3 {
bool prime(uint64_t n);
int64_t nearestPrime(int n, double bits, std::set<int64_t> &used);
} // namespace snabf::pass3

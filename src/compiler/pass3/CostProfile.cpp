#include "CostProfile.h"
#include "PrimeInstantiator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <charconv>
#include <set>

namespace snabf::pass3 {
namespace {
using namespace llvm::json;
void need(bool ok, std::string_view why) {
  if (!ok)
    throw std::invalid_argument("invalid_cost_profile: " + std::string(why));
}
const Value &at(const Object &o, llvm::StringRef k) {
  auto *v = o.get(k);
  need(v != nullptr, k.str());
  return *v;
}
const Object &object(const Value &v) {
  auto *o = v.getAsObject();
  need(o, "expected object");
  return *o;
}
const Array &array(const Value &v) {
  auto *a = v.getAsArray();
  need(a, "expected array");
  return *a;
}
std::string text(const Value &v) {
  auto s = v.getAsString();
  need(s && !s->empty(), "expected nonempty string");
  return s->str();
}
int64_t integer(const Value &v, int64_t lo, int64_t hi) {
  auto n = v.getAsInteger();
  need(n && *n >= lo && *n <= hi, "integer out of range");
  return *n;
}
bool hex(const std::string &s, size_t n) {
  return s.size() == n && s.find_first_not_of("0123456789abcdef") == s.npos;
}

// Canonical integer JSON tree, shared with the Go writer. Count keys as well:
// JSON parsers may silently overwrite duplicate fields before validation.
void canonical(const Value &v, std::string &s, size_t &keys) {
  auto str = [&](llvm::StringRef t) {
    s += 's';
    s += std::to_string(t.size());
    s += ':';
    s.append(t.data(), t.size());
  };
  if (auto *o = v.getAsObject()) {
    s += '{';
    std::vector<llvm::StringRef> names;
    for (auto &it : *o)
      names.push_back(it.first);
    std::sort(names.begin(), names.end());
    keys += names.size();
    for (auto k : names) {
      str(k);
      canonical(*o->get(k), s, keys);
    }
    s += '}';
  } else if (auto *a = v.getAsArray()) {
    s += '[';
    for (auto &x : *a)
      canonical(x, s, keys);
    s += ']';
  } else if (auto t = v.getAsString())
    str(*t);
  else if (auto i = v.getAsInteger()) {
    s += 'i';
    s += std::to_string(*i);
    s += ';';
  } else
    need(false, "non-integer/unsupported JSON value");
}
size_t scan(std::string_view s) {
  bool quoted = false, escape = false;
  size_t keys = 0;
  int depth = 0;
  for (char c : s) {
    if (quoted) {
      if (escape)
        escape = false;
      else if (c == '\\')
        escape = true;
      else if (c == '"')
        quoted = false;
      continue;
    }
    if (c == '"')
      quoted = true;
    else if (c == ':')
      ++keys;
    else if (c == '{' || c == '[')
      need(++depth <= 32, "nesting");
    else if (c == '}' || c == ']')
      --depth;
    else
      need(c != '.' && c != 'e' && c != 'E',
           "numbers must be decimal integers");
  }
  return keys;
}
std::string digest(const Value &v, size_t &keys) {
  std::string s;
  canonical(v, s, keys);
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(s)), true);
}

// Parse fixed five-place decimal microseconds into quarter-nanoseconds.
int64_t ticks(const Value &v) {
  std::string t = text(v);
  auto dot = t.find('.');
  need(dot >= 1 && dot <= 10 && t.size() == dot + 6 &&
           (dot == 1 || t[0] != '0'),
       "microsecond decimal");
  auto dec = [](std::string_view s) {
    int64_t x = 0;
    need(s.find_first_not_of("0123456789") == s.npos, "unsigned microseconds");
    auto [p, e] = std::from_chars(s.data(), s.data() + s.size(), x);
    need(e == std::errc{} && p == s.data() + s.size(),
         "microseconds out of range");
    return x;
  };
  auto whole = dec(std::string_view(t).substr(0, dot)),
       frac = dec(std::string_view(t).substr(dot + 1));
  need(frac % 25 == 0, "sub-tick time");
  return whole * 4000 + frac / 25;
}
std::vector<int64_t> samples(const Value &v, int count) {
  std::vector<int64_t> a;
  need(int(array(v).size()) == count, "sample count");
  for (auto &s : array(v)) {
    auto n = ticks(s);
    need(n > 0 && n % 4 == 0, "nonpositive or fractional-ns raw sample");
    a.push_back(n / 4);
  }
  return a;
}
std::vector<int> ints(const Value &v, int lo, int hi) {
  std::vector<int> a;
  for (auto &x : array(v))
    a.push_back(integer(x, lo, hi));
  return a;
}
} // namespace

std::shared_ptr<const CostProfile> CostProfile::loadFile(std::string_view path,
                                                         std::string_view hw) {
  auto f = llvm::MemoryBuffer::getFile(llvm::StringRef(path));
  need(bool(f), "cannot read file");
  return parse((*f)->getBuffer().str(), hw);
}
std::shared_ptr<const CostProfile>
CostProfile::parse(std::string_view raw, std::string_view expected) {
  need(raw.size() <= 64 * 1024 * 1024, "file too large");
  size_t rawKeys = scan(raw);
  auto parsed = llvm::json::parse(llvm::StringRef(raw));
  if (!parsed)
    throw std::invalid_argument("invalid_cost_profile: " +
                                llvm::toString(parsed.takeError()));
  auto *o = parsed->getAsObject();
  need(o, "profile object");
  auto p = std::shared_ptr<CostProfile>(new CostProfile);
  p->digest_ = text(at(*o, "profile_sha256"));
  o->erase("profile_sha256");
  size_t keys = 1;
  need(hex(p->digest_, 64) &&
           ::snabf::pass3::digest(*parsed, keys) == p->digest_,
       "profile_sha256");
  need(keys == rawKeys, "duplicate object key");
  need(text(at(*o, "schema")) == "snabf-gadget-sites-v1" &&
           text(at(*o, "abi")) == "direct-go-expanded-pure-rns-ntt-v1" &&
           text(at(*o, "units")) == "us",
       "unknown schema, units or ABI");
  p->hardware_ = text(at(*o, "hardware_id"));
  auto &hv = at(*o, "hardware");
  auto &h = object(hv);
  keys = 0;
  need(hex(p->hardware_, 64) && p->hardware_ == expected &&
           ::snabf::pass3::digest(hv, keys) == p->hardware_,
       "hardware_id mismatch");
  auto scope = text(at(h, "id_scope"));
  need(scope == "machine-id" || scope == "boot-id",
       "unknown hardware ID scope");
  for (auto k : {"arch", "cpu_model", "microcode", "governor"})
    text(at(h, k));
  auto affinity = text(at(h, "affinity"));
  need(affinity.find_first_not_of("0123456789") == affinity.npos &&
           text(at(h, "os")) == "linux" &&
           hex(text(at(h, "machine_sha256")), 64),
       "hardware metadata");
  auto &tc = object(at(*o, "toolchain"));
  for (auto k : {"go", "build"})
    text(at(tc, k));
  need(text(at(tc, "gomaxprocs")) == "1" &&
           text(at(tc, "gc")) ==
               "forced-before-each-call; disabled-inside-clock" &&
           text(at(tc, "lattigo")) == "v6.2.0" &&
           hex(text(at(tc, "binary_sha256")), 64),
       "toolchain metadata");
  need(hex(text(at(*o, "bench_rev")), 40) &&
           text(at(*o, "quantiles")) == "type7-from-raw-us",
       "revision or quantile convention");
  text(at(*o, "created_utc"));
  int nSamples = integer(at(*o, "samples"), 1, 10000),
      hold = integer(at(*o, "holdout"), 1, 10000);
  integer(at(*o, "warmup"), 1, 10000);
  for (auto &v : array(at(*o, "q_p_shape"))) {
    const auto &b = object(v);
    int n = integer(at(b, "n"), 16, 65536);
    need(!(n & (n - 1)), "non-power-of-two N");
    MeasuredBasis shape{n, {}, {}, {}};
    std::set<int64_t> seen;
    for (auto k : {"q", "p"}) {
      auto &dest = (k[0] == 'q' ? shape.q : shape.p);
      auto &xs = array(at(b, k));
      need(!xs.empty() && xs.size() <= (k[0] == 'q' ? 64u : 4u), "basis shape");
      for (auto &x : xs) {
        int64_t z = integer(x, 3, (1LL << 60) - 1);
        need(z % (2LL * n) == 1 && prime(z) && seen.insert(z).second,
             "invalid ordered NTT basis");
        dest.push_back(z);
      }
    }
    for (auto &b0 : p->bases_)
      need(n != b0.n || shape.q != b0.q || shape.p != b0.p, "duplicate basis");
    p->bases_.push_back(std::move(shape));
  }
  need(!p->bases_.empty(), "empty basis grid");
  for (auto &v : array(at(*o, "sites"))) {
    auto &row = object(v);
    auto &b = p->bases_.at(integer(at(row, "shape"), 0, p->bases_.size() - 1));
    int alpha = b.p.size(), ell = integer(at(row, "ell"), 0, b.q.size() - 1);
    auto op = text(at(row, "op"));
    need(op == "relin" || op == "giant" || op == "baby", "unknown call");
    auto shifts = ints(at(row, "shifts"), 1, b.n / 2 - 1);
    auto ks = ints(at(row, "k"), 0, std::min(ell, 4));
    need(std::set<int>(shifts.begin(), shifts.end()).size() == shifts.size() &&
             (op == "relin" ? shifts.empty() && ks.size() == 1
                            : !shifts.empty() && shifts.size() == ks.size() &&
                                  (op != "giant" || ks.size() == 1)),
         "call arity/shift");
    std::vector<int> rows;
    for (int j = 0; j <= ell; j += alpha)
      rows.push_back(std::min(alpha, ell + 1 - j));
    need(ints(at(row, "block_rows"), 1, alpha) == rows, "original block rows");
    const auto &obs = array(at(row, "observed_non_native"));
    size_t j = 0;
    for (int k : ks)
      if (k) {
        need(j < obs.size(), "missing diagnostic");
        auto &e = object(obs[j++]);
        need(integer(at(e, "truncated_limbs_k"), 0, ell) == k &&
                 integer(at(e, "full_blocks_skipped"), 0, ell + 1) == k / alpha,
             "inconsistent observed partition");
        integer(at(e, "measured_single_limb_ntts_saved"), INT32_MIN, INT32_MAX);
      }
    need(j == obs.size(), "unexpected diagnostic");
    auto a = samples(at(row, "samples_us"), nSamples);
    samples(at(row, "holdout_us"), hold);
    std::sort(a.begin(), a.end());
    auto q = [&](int j) {
      size_t x = (a.size() - 1) * j, i = x / 4, r = x % 4;
      return int64_t((4 - r) * a[i] + r * a[std::min(i + 1, a.size() - 1)]);
    };
    int64_t med = q(2);
    need(ticks(at(row, "median_us")) == med &&
             ticks(at(row, "iqr_us")) == q(3) - q(1),
         "raw samples disagree with quantiles");
    CostCall call{ell,
                  op == "relin"  ? Op::Relin
                  : op == "baby" ? Op::Baby
                                 : Op::Giant,
                  std::move(shifts)};
    need(b.calls[call].emplace(ks, med / 4000.).second, "duplicate k pattern");
  }
  for (auto &b : p->bases_) {
    need(!b.calls.empty(), "basis has no observations");
    for (auto &[call, ps] : b.calls)
      if (!ps.contains(
              std::vector<int>(std::max(size_t(1), call.shifts.size()), 0)))
        throw MissingCost("all-native pattern");
  }
  p->json_ = raw;
  return p;
}
} // namespace snabf::pass3

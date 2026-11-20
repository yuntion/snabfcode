#include "AffineErrorForm.h"

#include <cmath>
#include <mpfr.h>
#include <stdexcept>

namespace snabf::pass2 {
namespace {
struct Float {
  mpfr_t x;
  explicit Float(mpfr_prec_t p = 256) { mpfr_init2(x, p); }
  ~Float() { mpfr_clear(x); }
  Float(const Float &) = delete;
  Float &operator=(const Float &) = delete;
};

void require(bool ok, const char *why) {
  if (!ok) throw std::invalid_argument(why);
}

// mpq_class(num, den) does not canonicalize. Arithmetic on malformed
// public inputs (in particular a zero denominator) is not a GMP contract.
void checkRational(const Exact &q) {
  auto den = mpq_denref(q.get_mpq_t());
  require(mpz_sgn(den) > 0, "invalid rational denominator");
  if (mpz_cmp_ui(den, 1) != 0) {
    mpz_class gcd;
    mpz_gcd(gcd.get_mpz_t(), mpq_numref(q.get_mpq_t()), den);
    require(gcd == 1, "noncanonical rational bound");
  }
}

Exact rational(const Float &v) {
  if (!mpfr_number_p(v.x)) throw std::overflow_error("non-finite MPFR result");
  Exact r;
  mpfr_get_q(r.get_mpq_t(), v.x);
  return r;
}

double asDouble(const Exact &q, mpfr_rnd_t dir) {
  checkRational(q);
  Float t;
  mpfr_set_q(t.x, q.get_mpq_t(), dir);
  double v = mpfr_get_d(t.x, dir);
  if (!std::isfinite(v)) throw std::overflow_error("unrepresentable audit bound");
  return v;
}
} // namespace

Exact exactBinary(double value) {
  require(std::isfinite(value), "expected a finite binary64 value");
  return Exact(value);
}
double upperDouble(const Exact &q) { return asDouble(q, MPFR_RNDU); }
double lowerDouble(const Exact &q) { return asDouble(q, MPFR_RNDD); }

Exact sqrtUpper(const Exact &q) {
  checkRational(q);
  require(q >= 0, "negative squared bound");
  Float t;
  mpfr_set_q(t.x, q.get_mpq_t(), MPFR_RNDU);
  mpfr_sqrt(t.x, t.x, MPFR_RNDU);
  return rational(t);
}

Exact logUpper(const Exact &q) {
  checkRational(q);
  require(q > 0, "invalid logarithm argument");
  Float t;
  mpfr_set_q(t.x, q.get_mpq_t(), MPFR_RNDU);
  mpfr_log(t.x, t.x, MPFR_RNDU);
  return rational(t);
}

Exact roundCKKSScale(const Exact &q) {
  checkRational(q);
  require(q > 0, "non-positive CKKS scale");
  Float t(128);
  mpfr_set_q(t.x, q.get_mpq_t(), MPFR_RNDN);
  // Bound resource use on repeated ciphertext squaring. Go's exponent
  // range is wider; out-of-scope scales fail closed, never saturate.
  require(mpfr_number_p(t.x) && !mpfr_zero_p(t.x) &&
          mpfr_get_exp(t.x) >= -16384 && mpfr_get_exp(t.x) <= 16384,
          "physical scale outside replay range");
  return rational(t);
}

void AffineErrorForm::addBias(const Exact &b) {
  checkRational(b);
  require(b >= 0, "negative bias");
  bias_ += b;
}

void AffineErrorForm::addLinear(const std::string &source, const LinearTerm &t) {
  checkRational(t.signed_coeff);
  checkRational(t.base_var_proxy);
  require(!source.empty() && !t.formula_ref.empty() && t.base_var_proxy >= 0,
          "invalid tracked source");
  switch (t.kind) {
  case NoiseSourceKind::FreshEncrypt: break;
  case NoiseSourceKind::FreshEncode:
    require(t.joint_proof.empty(), "encoding is not a fresh independent draw");
    break;
  case NoiseSourceKind::ProvenIndependent:
    require(!t.joint_proof.empty(), "unproved independent source");
    break;
  default: throw std::invalid_argument("unknown source kind");
  }
  auto [it, inserted] = linear_.try_emplace(source, t);
  if (!inserted) {
    auto &a = it->second;
    require(a.kind == t.kind && a.base_var_proxy == t.base_var_proxy &&
            a.formula_ref == t.formula_ref && a.joint_proof == t.joint_proof,
            "conflicting definitions of one random variable");
    a.signed_coeff += t.signed_coeff;
  }
}

void AffineErrorForm::addDependent(const DependentNoiseTerm &t) {
  checkRational(t.sigma_proxy);
  require(!t.site_id.empty() && !t.formula_ref.empty() && t.sigma_proxy >= 0,
          "invalid dependent residual");
  require(t.kind != DependentKind::SharedEvkSwitch || !t.shared_key_id.empty(),
          "missing evaluation key identity");
  auto [it, inserted] = dependent_.try_emplace(t.site_id, t);
  if (!inserted) {
    auto &a = it->second;
    require(a.kind == t.kind && a.shared_key_id == t.shared_key_id &&
            a.formula_ref == t.formula_ref, "conflicting dependent site");
    a.sigma_proxy += t.sigma_proxy;
  }
}

void AffineErrorForm::addEvent(const EventKey &key, const Exact &p) {
  checkRational(p);
  require(!key.first.empty() && !key.second.empty() && p >= 0 && p < 1,
          "invalid common-event term");
  auto [it, ok] = events_.emplace(key, p);
  require(ok || it->second == p, "conflicting bounds for one event");
}

void AffineErrorForm::markMissing(const std::string &site) {
  require(!site.empty(), "missing contract without a site");
  missing_.insert(site);
}

void AffineErrorForm::inheritObligations(const AffineErrorForm &a) {
  if (this == &a) return;
  for (const auto &[key, p] : a.events_) addEvent(key, p);
  missing_.insert(a.missing_.begin(), a.missing_.end());
}

void AffineErrorForm::addScaled(const AffineErrorForm &a, const Exact &gain) {
  checkRational(gain);
  if (this == &a) { auto copy = a; addScaled(copy, gain); return; }
  // gain may be a reference into either form.
  const Exact k = gain, magnitude = abs(k);
  bias_ += magnitude * a.bias_;
  for (auto [id, t] : a.linear_) {
    t.signed_coeff *= k;
    addLinear(id, t);
  }
  for (auto [id, t] : a.dependent_) {
    t.sigma_proxy *= magnitude;
    addDependent(t);
  }
  inheritObligations(a); // Even a zero multiplier does not undo an event.
}

Exact AffineErrorForm::varianceUpper() const {
  // Different joint proofs do NOT establish independence across groups.
  // Separate key spaces also prevent a source ID from naming a proof family.
  std::map<std::pair<bool, std::string>, Exact> groups;
  for (const auto &[id, t] : linear_) {
    if (t.signed_coeff == 0 || t.base_var_proxy == 0) continue;
    bool single = t.joint_proof.empty();
    groups[{single, single ? id : t.joint_proof}] +=
        t.signed_coeff * t.signed_coeff * t.base_var_proxy;
  }
  Exact sigma = 0;
  for (const auto &[id, d] : dependent_) sigma += d.sigma_proxy;
  if (sigma == 0 && groups.size() == 1) return groups.begin()->second;
  for (const auto &[key, v] : groups) sigma += sqrtUpper(v);
  return sigma * sigma;
}

Exact AffineErrorForm::deltaValid() const {
  Exact sum = 0;
  for (const auto &[key, p] : events_) sum += p;
  return sum;
}
} // namespace snabf::pass2

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <span>
#include <vector>

namespace snabf::ir::affine {

struct Interval {
  double lo, hi;
  bool operator==(const Interval&) const = default;
};

// Deterministic message values, before CKKS arithmetic. A symbol denotes a
// real in [-1,1], not a random variable. The remainder has no cancellable ID.
struct AffineRange {
  double center = 0;
  std::map<std::uint64_t, double> coefficients;
  double remainder = 0;
  Interval toSoundIntervalMPFR(unsigned precision = 160) const;
};

struct SoundValue {
  AffineRange form;
  Interval naive, enclosure;  // Independent natural extension; proven intersection.
  std::size_t condensed = 0;  // This transfer only.
  double roundingRadius = 0; // Local center/coefficient storage error, in message units.
};

enum class Activation { SiLU, ErfGeLU, ReLU };
enum class NonlinearMode { AffineResidual, IndependentInterval };

class AffineIntervalAnalyzer {
 public:
  using Slots = std::vector<SoundValue>;
  class Value {
    friend class AffineIntervalAnalyzer;
    const AffineIntervalAnalyzer* owner_;
    std::size_t index_;
    Value(const AffineIntervalAnalyzer* owner, std::size_t index)
        : owner_(owner), index_(index) {}
   public:
    bool operator==(const Value&) const = default;
  };

  explicit AffineIntervalAnalyzer(unsigned precision = 160, std::size_t maxSymbols = 256);
  AffineIntervalAnalyzer(const AffineIntervalAnalyzer&) = delete;
  AffineIntervalAnalyzer& operator=(const AffineIntervalAnalyzer&) = delete;

  // A call declares a distinct input port. Reuse its returned SSA value on
  // every fanout, including both arms of a residual connection.
  Value source(std::span<const Interval> domains);
  Value add(Value a, Value b);
  Value subtract(Value a, Value b);
  Value negate(Value x);
  // Row-major W[output,input]; bias is absent or one value per output.
  Value linear(Value x, std::size_t outputs, std::span<const double> weights,
               std::span<const double> bias = {});
  // table[d, i] multiplies x[(i+offset[d]) % period]. Input tiling must
  // already be proven; here every physical row replica is checked explicitly.
  Value periodicDiagonals(Value x, std::size_t physicalPeriod,
                         std::span<const std::int64_t> offsets,
                         std::span<const double> table);
  // Includes the ideal output AND any real implementation satisfying
  // |p(t)-f(t)| <= certifiedError on the entire incoming enclosure. An error
  // budget is sufficient for a family only if admission checks every member.
  Value activate(Value x, Activation f, double certifiedError = 0,
                 NonlinearMode mode = NonlinearMode::AffineResidual);

  const Slots& slots(Value v) const;
  Interval domain(Value v) const;
  unsigned precision() const { return precision_; }
  std::size_t symbolLimit() const { return maxSymbols_; }

 private:
  Value store(Slots slots);
  Value binary(Value a, Value b, bool subtract);
  std::uint64_t fresh();
  unsigned precision_;
  std::size_t maxSymbols_;
  std::uint64_t next_ = 0;
  std::deque<Slots> nodes_;
};

}  // namespace snabf::ir::affine

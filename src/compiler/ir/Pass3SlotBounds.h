#pragma once

#include "Pass2IR.h"
#include "pass3/GadgetNoiseContract.h"
#include <algorithm>
#include <cmath>

namespace snabf::ir {
// Decoded *ideal* absolute bounds for a single proven periodic packing. Store
// the whole encoded period, not a logical-width prefix: absent an affine
// replica proof, two table columns with the same logical index may differ.
class Pass3SlotBounds {
public:
  struct Operands {
    int64_t shift = 0;        // BSGS rotation, or the mask's giant offset
    ArrayRef<double> row;     // unshifted prepared diagonal in output coordinates
    double absConstant = 0;   // add_plain's actual literal, not a difference
  };

  explicit Pass3SlotBounds(size_t period) : period(period) {
    require(period > 0, "empty Pass 3 slot period");
  }

  // Require the same row binding and physical decomposition as the emitter.
  // A positional association of an unsorted table with a sorted route is not
  // a proof; reject rather than assign a small mask to the wrong rotation.
  void checkTerm(int64_t tableOffset, int64_t d, int64_t b, int64_t g,
                 int64_t group) const {
    require(group > 0 && group % int64_t(period) == 0 && tableOffset == d &&
                d >= 0 && d < int64_t(period) && b >= 0 && b < group &&
                g >= 0 && g < group && (b + g) % group == d,
            "unproven Pass 3 BSGS row/rotation binding");
  }

  double append(const pass3::Edge &e, Operands t) {
    using O = pass3::Op;
    std::vector<double> out(period);
    auto parent = [&](size_t k) -> const std::vector<double> & {
      return slots.at(e.inputs.at(k));
    };
    switch (e.op) {
    case O::Source:
      require(std::isfinite(e.magnitude) && e.magnitude >= 0, "invalid Pass 3 source bound");
      std::fill(out.begin(), out.end(), e.magnitude);
      break;
    case O::Baby: case O::Giant: {
      const auto &a = parent(0);
      size_t shift = size_t(t.shift) % period;
      for (size_t i = 0; i < period; ++i) out[i] = a[(i + shift) % period];
      break;
    }
    case O::Mask: {
      require(t.row.size() == period, "unbound Pass 3 diagonal width");
      const auto &baby = parent(0);
      size_t giant = size_t(t.shift) % period;
      // Rot_g(Rot_-g(D_d) * Rot_b(x))[i] = D_d[i] * x[i+b+g].
      for (size_t i = 0; i < period; ++i)
        out[i] = upperProduct(std::abs(t.row[(i + period - giant) % period]), baby[i]);
      break;
    }
    case O::Add:
      for (int a : e.inputs)
        for (size_t i = 0; i < period; ++i) out[i] = upperSum(out[i], slots.at(a)[i]);
      break;
    case O::Multiply: {
      const auto &a = parent(0), &b = parent(1);
      for (size_t i = 0; i < period; ++i) out[i] = upperProduct(a[i], b[i]);
      break;
    }
    case O::Coefficient: case O::AddPlain: {
      const auto &a = parent(0);
      for (size_t i = 0; i < period; ++i)
        out[i] = e.op == O::Coefficient ? upperProduct(a[i], e.gains.at(0))
                                        : upperSum(a[i], t.absConstant);
      break;
    }
    case O::Drop: case O::Relin: case O::Rescale:
      out = parent(0); // scales and arithmetic error are separate recurrences
      break;
    default:
      throw ContractError("unproven Pass 3 slot transfer");
    }
    double peak = *std::max_element(out.begin(), out.end());
    slots.push_back(std::move(out));
    return peak;
  }

  // Only the completed SSA value has a Pass 2 enclosure. Never cap a partial
  // sum, a polynomial intermediate, or another operand with its result's fact.
  void finish(size_t first, size_t result, pass3::Edge &e, ArrayRef<double> affine) {
    if (!affine.empty() && result >= first) {
      require(period % affine.size() == 0, "invalid Pass 3 affine slot period");
      require(llvm::all_of(affine, [](double b) { return std::isfinite(b) && b >= 0; }),
              "invalid Pass 3 affine magnitude");
      auto &v = slots.at(result);
      for (size_t i = 0; i < period; ++i) v[i] = std::min(v[i], affine[i % affine.size()]);
      e.magnitude = *std::max_element(v.begin(), v.end());
    }
    // Only SSA results can be named by subsequent expanded operations.
    for (size_t i = first; i < slots.size(); ++i)
      if (i != result) std::vector<double>().swap(slots[i]);
  }

private:
  size_t period;
  std::vector<std::vector<double>> slots;
};

// A separate unit box for the entrance error of ONE lintrans. Not an ideal
// magnitude, not intersected with affine facts (ideal-zero lanes can be noisy).
class LinearEntryBounds {
public:
  LinearEntryBounds(size_t period, int entry, int first)
      : unit(period), entry(entry), first(first), next(first) {
    require(entry >= 0 && entry < first, "invalid lintrans entrance");
    pass3::Edge source{"unit-error", pass3::Op::Source, 0, {}, {}, 1.};
    unit.append(source, {});
  }

  void annotate(int index, pass3::Edge &edge, Pass3SlotBounds::Operands t) {
    using O = pass3::Op;
    require(index == next && !edge.inputs.empty() &&
                (edge.op == O::Mask || edge.op == O::Baby || edge.op == O::Giant ||
                 edge.op == O::Add || edge.op == O::Rescale),
            "unproven lintrans noise transfer");
    auto local = edge;
    for (int &a : local.inputs) {
      require(a == entry || (a >= first && a < index), "foreign lintrans error input");
      a = a == entry ? 0 : 1 + a - first;
    }
    edge.entryGain = unit.append(local, t);
    edge.linearEntry = entry;
    ++next;
  }

private:
  Pass3SlotBounds unit;
  int entry, first, next;
};
} // namespace snabf::ir

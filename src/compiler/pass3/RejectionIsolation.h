#pragma once
#include "GadgetNoiseContract.h"
#include "Highs.h"
#include <boost/property_tree/ptree.hpp>

namespace snabf::pass3::diagnostic {
using Tree = boost::property_tree::ptree;

template <typename T> inline void append(Tree &a, T x) {
  Tree item;
  item.put_value(x);
  a.push_back({"", std::move(item)});
}

// Keep the actual DAG, not a reported scale class or a hand-picked edge.
inline Tree baseWidth(const Graph &g, const Options &o, double lower) {
  Tree t, dag;
  t.put("kind", "base_width_v1");
  t.put("min_base", o.minBase);
  t.put("guard_bits", o.guardBits);
  t.put("exclusive_ceiling", o.blockSearch() ? o.primeWidthMax : 60.);
  t.put("builder_lower", lower);
  for (const auto &e : g.edges) {
    Tree v, inputs;
    v.put("id", e.id);
    v.put("op", name(e.op));
    v.put("level", e.level);
    if (e.level == 0) v.put("magnitude", e.magnitude);
    for (int i : e.inputs) append(inputs, i);
    if (!inputs.empty()) v.add_child("inputs", std::move(inputs));
    dag.push_back({"", std::move(v)});
  }
  t.add_child("edges", std::move(dag));
  return t;
}

// Original model of this round, including refinement cuts; never mutate or
// re-solve it for diagnosis. Column bounds and coefficients round-trip to f64.
inline Tree solverRows(const Highs &h, int qpRow) {
  const auto &lp = h.getLp();
  auto a = lp.a_matrix_;
  a.ensureRowwise();
  Tree t, columns, rows;
  t.put("kind", "individual_lp_rows_v1");
  t.put("highs_model_status", h.modelStatusToString(h.getModelStatus()));
  t.put("num_cols", lp.num_col_);
  t.put("num_rows", lp.num_row_);
  t.put("qp_cap_row", qpRow);
  auto bounds = [](double l, double u) {
    Tree pair;
    if (l == -kHighsInf) append(pair, "-inf"); else append(pair, l);
    if (u == kHighsInf) append(pair, "inf"); else append(pair, u);
    return pair;
  };
  for (HighsInt j = 0; j < lp.num_col_; ++j)
    columns.push_back({"", bounds(lp.col_lower_[j], lp.col_upper_[j])});
  for (HighsInt i = 0; i < lp.num_row_; ++i) {
    auto row = bounds(lp.row_lower_[i], lp.row_upper_[i]);
    for (HighsInt k = a.start_[i]; k < a.start_[i + 1]; ++k) {
      append(row, a.index_[k]);
      append(row, a.value_[k]);
    }
    rows.push_back({"", std::move(row)});
  }
  t.add_child("columns", std::move(columns));
  t.add_child("rows", std::move(rows));
  return t;
}
} // namespace snabf::pass3::diagnostic

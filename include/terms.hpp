// Copyright 2021 Pierre Talbot, Frédéric Pinel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TERMS_HPP
#define TERMS_HPP

#include "cuda_helper.hpp"

// NOTE: Bitwise OR and AND are necessary to avoid short-circuit of Boolean operators.

class Constant {
  int c;
public:
  typedef Constant neg_type;
  Constant(int c) : c(c) {}
  bool update_lb(VStore&, int) { return false; }
  bool update_ub(VStore&, int) { return false; }
  int lb(VStore&) { return c; }
  int ub(VStore&) { return c; }
  bool is_top(VStore&) { return false; }
  neg_type neg() { return Constant(-c); }
  void print(VStore&) { printf("%d", c); }
};

template <typename Term>
class Negation {
  Term t;
public:
  typedef Term neg_type;
  Negation(Term t) : t(t) {}
  bool update_lb(VStore& vstore, int lb) {
    return t.update_ub(vstore, -lb);
  }
  bool update_ub(VStore& vstore, int ub) {
    return t.update_lb(vstore, -ub);
  }
  int lb(VStore& vstore) { return -t.ub(vstore); }
  int ub(VStore& vstore) { return -t.lb(vstore); }
  bool is_top(VStore& vstore) { return t.is_top(vstore); }
  neg_type neg() { return t; }
  void print(VStore& vstore) { printf("-"); t.print(vstore); }
};

class Variable {
  int idx;
public:
  typedef Negation<Variable> neg_type;
  Variable(int idx) : idx(idx) {}
  bool update_lb(VStore& vstore, int lb) {
    return vstore.update_lb(idx, lb);
  }
  bool update_ub(VStore& vstore, int ub) {
    return vstore.update_ub(idx, ub);
  }
  int lb(VStore& vstore) { return vstore.lb(idx); }
  int ub(VStore& vstore) { return vstore.ub(idx); }
  bool is_top(VStore& vstore) { return vstore.is_top(idx); }
  neg_type neg() { return neg_type(*this); }
  void print(VStore& vstore) { vstore.print_var(idx); }
};

template<typename Term>
class Absolute {
  Term t;
public:
  typedef Negation<Absolute<Term>> neg_type;
  Absolute(Term t): t(t) {}

  // |t| >= k
  bool update_lb(VStore& vstore, int k) {
    bool has_changed = false;
    if(k < 0) has_changed |= t.update_lb(limit_max());
    if(t.lb(vstore) >= 0) has_changed |= t.update_lb(k);
    if(t.ub(vstore) < k) has_changed |= t.update_ub(-k);
    return has_changed;
  }

  // |t| <= k
  bool update_ub(VStore& vstore, int ub) {
    bool has_changed = false;
    if(k < 0) has_changed |= t.update_lb(limit_max());
    if(t.lb(vstore) < 0) has_changed |= t.update_lb(-k);
    if(t.ub(vstore) > k) has_changed |= t.update_ub(k);
    return has_changed;
  }

  int lb(VStore& vstore) {
    return min(abs(t.lb(vstore)), abs(t.ub(vstore)));
  }

  int ub(VStore& vstore) {
    return max(abs(t.lb(vstore)), abs(t.ub(vstore)));
  }

  bool is_top(VStore& vstore) { return t.is_top(vstore); }

  neg_type neg() { return neg_type(*this); }

  void print(VStore& vstore) {
    printf("|");
    t.print(vstore);
    printf("|");
  }
}

template <typename TermX, typename TermY>
class Add {
  TermX x;
  TermY y;
public:
  typedef Add<TermX::neg_type, TermY::neg_type> neg_type;
  Add(TermX x, TermY y) : x(x), y(y) {}

  // Enforce x + y >= k
  bool update_lb(VStore& vstore, int k) {
    return x.update_lb(vstore, k - y.ub(vstore)) |
           y.update_lb(vstore, k - x.ub(vstore));
  }

  // Enforce x + y <= k
  bool update_ub(VStore& vstore, int k) {
    return x.update_ub(vstore, k - y.u(vstore)) |
           y.update_ub(vstore, k - x.lb(vstore));
  }

  int lb(VStore& vstore) { return x.lb(vstore) + y.lb(vstore); }
  int ub(VStore& vstore) { return x.ub(vstore) + y.ub(vstore); }

  bool is_top(VStore& vstore) { return x.is_top(vstore) || y.is_top(vstore); }

  neg_type neg() { return neg_type(t.neg(), -k); }

  void print(VStore& vstore) {
    x.print(vstore);
    printf(" + ");
    y.print(vstore);
  }
};

int div_up(int a, int b) {
  assert(b != 0);
  int r = a / b;
  // division is rounded towards zero.
  // We add one only if `r` was truncated and `a, b` are of equal sign (so the division operated in the positive numbers).
  // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
  return (a % b != 0 && a > 0 == b > 0) ? r + 1 : r;
}

int div_down(int a, int b) {
  assert(b != 0);
  int r = a / b;
  return (a % b != 0 && a > 0 != b > 0) ? r - 1 : r;
}

template <typename TermX, typename TermY>
class Mul {
public:
  typedef Mul<TermX::neg_type, TermY> neg_type;
  neg_type neg() { return neg_type(t.neg(), u); }
private:
  TermX x;
  TermY y;
  neg_type neg_mul;
public:
  Mul(TermX x, TermY y) : x(x), y(y), neg_mul(neg()) {}

  // Enforce x * y >= k
  bool update_lb(VStore& vstore, int k) {
    // x * y >= k <=> -x * y <= -k - 1
    neg_mul.update_ub(vstore, -k - 1);
  }

  // Enforce x * y <= k
  bool update_ub(VStore& vstore, int k) {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    bool has_changed = false;
    if(k >= 0) {
      // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
      // When the sign are reversed, e.g. -x and y, or x and -y, these rules will automatically fails one of the domain.
      if(ux < 0) { has_changed |= y.update_ub(vstore, -1); }
      if(uy < 0) { has_changed |= x.update_ub(vstore, -1); }
      if(lx >= 0) { has_changed |= y.update_lb(vstore, 0); }
      if(ly >= 0) { has_changed |= x.update_lb(vstore, 0); }
      // Both signs are positive.
      if(lx > 0 && ly >= 0) { has_changed |= y.update_ub(vstore, k / lx); }
      if(lx >= 0 && ly > 0) { has_changed |= x.update_ub(vstore, k / ly); }
      // Both signs are negative.
      if(ux < 0 && uy < 0) {
        has_changed |= y.update_ub(vstore, div_up(k, lx));
        has_changed |= x.update_ub(vstore, div_up(k, ly));
      }
    }
    else {
      // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
      if(ux < 0) { has_changed |= y.update_lb(vstore, 0); }
      if(uy < 0) { has_changed |= x.update_lb(vstore, 0); }
      if(lx >= 0) { has_changed |= y.update_ub(vstore, -1); }
      if(ly >= 0) { has_changed |= x.update_ub(vstore, -1); }
      // When both variables have both signs.
      if(lx < 0 && ux > 0 && ly < 0 && uy > 0) {
        if(uy * lx > k) { has_changed |= x.update_ub(vstore, -1); }
        if(ux * ly > k) { has_changed |= y.update_ub(vstore, -1); }
      }
      // When the sign are reversed, e.g. -x and y, or x and -y.
      if(ux < 0 && uy > 0) { has_changed |= x.update_ub(div_up(k, uy)); }
      if(ux < 0 && uy >= 0) { has_changed |= y.update_lb(div_up(k, lx)); }
      if(ux >= 0 && uy < 0) { has_changed |= x.update_lb(div_up(k, ly)); }
      if(ux > 0 && uy < 0) { has_changed |= y.update_ub(div_up(k, ux)); }
    }
    return has_changed;
  }

  int lb(VStore& vstore) {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    return min(
      min(
        min(lx * ly),
        min(lx * uy)),
      min(
        min(ux * ly),
        min(ux * uy)));
  }

  int ub(VStore& vstore) {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    return max(
      max(
        max(lx * ly),
        max(lx * uy)),
      max(
        max(ux * ly),
        max(ux * uy)));
  }

  bool is_top(VStore& vstore) {
    return t.is_top(vstore) || u.is_top(vstore);
  }

  void print(VStore& vstore) {
    t.print(vstore);
    printf(" * ");
    u.print(vstore);
  }
};

// Just a simple checker, no propagation is performed.
// (yes, that could be improved!)
template<typename TermX, typename TermY>
class Modulo {
  TermX x;
  TermY y;
public:
  typedef Modulo<TermX::neg_type, TermY> neg_type;
  Modulo(TermX x, TermY y): x(x), y(y) {}

  // Enforce x % y >= k
  bool update_lb(VStore& vstore, int k) {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    if(lx == ux && ly == uy && lx % ly < k) {
      x.update_lb(limit_max());
      y.update_lb(limit_max());
      return true;
    }
    else {
      return false;
    }
  }

  // Enforce x % y <= k
  bool update_ub(VStore& vstore, int k) {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    if(lx == ux && ly == uy && lx % ly > k) {
      x.update_lb(limit_max());
      y.update_lb(limit_max());
      return true;
    }
    else {
      return false;
    }
  }

  int lb(VStore& vstore) {
    // [lx..ux] % [ly..uy] = [lr..ur]
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    int r = limit_max();
    for(int i = lx; i < lx + uy; ++i) {
      for(int j = ly; j < uy; ++j) {
        if(j != 0) r = min(r, i % j);
      }
    }
    return r;
  }
  int ub(VStore& vstore) {
    // [lx..ux] % [ly..uy] = [lr..ur]
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    int r = limit_max();
    for(int i = lx; i < lx + uy; ++i) {
      for(int j = ly; j < uy; ++j) {
        if(j != 0) r = max(r, i % j);
      }
    }
    return r;
  }

  bool is_top(VStore& vstore) { return x.is_top(vstore) || y.is_top(vstore); }

  neg_type neg() { return neg_type(x.neg(), y); }

  void print(VStore& vstore) {
    x.print(vstore);
    printf(" + ");
    y.print(vstore);
  }
}

namespace test {

}

#endif

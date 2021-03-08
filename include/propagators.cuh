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

#ifndef CONSTRAINTS_HPP
#define CONSTRAINTS_HPP

#include <cassert>
#include <cmath>
#include <stdexcept>
#include "cuda_helper.hpp"
#include "vstore.cuh"

class Propagator {
public:
  int uid;
public:
  Propagator() = default;
  CUDA Propagator(int uid): uid(uid) {}
  CUDA virtual ~Propagator() {}
  CUDA virtual bool propagate(VStore& vstore) const = 0;
  CUDA virtual bool is_entailed(const VStore& vstore) const = 0;
  CUDA virtual bool is_disentailed(const VStore& vstore) const = 0;
  CUDA virtual void print(const VStore& vstore) const = 0;
  virtual Propagator* neg() const = 0;
  // This function is called from host, and copy the object on the device memory.
  virtual Propagator* to_device() const = 0;
};

CUDA_GLOBAL void init_temporal_prop(Propagator** p, int uid, Var x, Var y, int c);
CUDA_GLOBAL void init_logical_or(Propagator** p, int uid, Propagator* left, Propagator* right);
CUDA_GLOBAL void init_logical_and(Propagator** p, int uid, Propagator* left, Propagator* right);
CUDA_GLOBAL void init_reified_prop(Propagator** p, int uid, Var b, Propagator* rhs, Propagator* not_rhs);
CUDA_GLOBAL void init_linear_ineq(Propagator** p, int uid, int n, const Var* vars, const int* constants, int max);


/// x + y <= c
template <typename X, typename Y, typename Z>
class XPlusYleqZ: public Propagator {

public:
  const X x;
  const Y y;
  const Z z;

  CUDA TemporalProp(X x, Y y, Z z):
   Propagator(-1), x(x), y(y), z(z)
  {
    assert(x != 0 && y != 0);
  }

  CUDA ~TemporalProp() {}

  CUDA bool propagate(VStore& vstore) const override {
    bool has_changedX = x.update_ub(vstore, z.ub(vstore) - y.lb(vstore));
    bool has_changedY = y.update_ub(vstore, z.ub(vstore) - x.lb(vstore));
    bool has_changedZ = z.update_lb(vstore, x.lb(vstore) + y.lb(vstore));
    return has_changedX || has_changedY || has_changedZ;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return
      !x.is_top(vstore) &&
      !y.is_top(vstore) &&
      !z.is_top(vstore) &&
      x.ub(vstore) + y.ub(vstore) <= z.lb(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return x.is_top(vstore) ||
           y.is_top(vstore) ||
           z.is_top(vstore) ||
           x.lb(vstore) + y.lb(vstore) > z.ub(vstore);
  }

  Propagator* neg() const {
    return new TemporalProp<X::neg_type, Y::neg_type, Add<Z::neg_type>>
      (x.neg(), y.neg(), Add(z.neg(), -1));
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    vstore.print_var(x);
    printf(" + ");
    vstore.print_var(y);
    printf(" <= %d", c);
  }

  Propagator* to_device() const override {
    Propagator** p;
    malloc2_managed(p, 1);
    init_temporal_prop<<<1, 1>>>(p, uid, x, y, c);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

/// x + y <= c
class TemporalProp: public Propagator {

public:
  const Var x;
  const Var y;
  const int c;

  CUDA TemporalProp(Var x, Var y, int c):
   Propagator(-1), x(x), y(y), c(c)
  {
    assert(x != 0 && y != 0);
  }

  CUDA ~TemporalProp() {}

  CUDA bool propagate(VStore& vstore) const override {
    bool has_changed = vstore.update_ub(x, c - vstore.lb(y));
    has_changed |= vstore.update_ub(y, c - vstore.lb(x));
    return has_changed;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return
      !vstore.is_top(x) &&
      !vstore.is_top(y) &&
      vstore.ub(x) + vstore.ub(y) <= c;
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return vstore.is_top(x) ||
           vstore.is_top(y) ||
           vstore.lb(x) + vstore.lb(y) > c;
  }

  Propagator* neg() const {
    return new TemporalProp(-x, -y, -c - 1);
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    vstore.print_var(x);
    printf(" + ");
    vstore.print_var(y);
    printf(" <= %d", c);
  }

  Propagator* to_device() const override {
    Propagator** p;
    malloc2_managed(p, 1);
    init_temporal_prop<<<1, 1>>>(p, uid, x, y, c);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

// C1 \/ C2
class LogicalOr: public Propagator {
public:
  const Propagator* left;
  const Propagator* right;

  CUDA LogicalOr(Propagator* left, Propagator* right):
    Propagator(-1), left(left), right(right) {}

  CUDA ~LogicalOr() {}

  CUDA bool propagate(VStore& vstore) const override {
    if (left->is_disentailed(vstore)) {
      return right->propagate(vstore);
    }
    else if (right->is_disentailed(vstore)) {
      return left->propagate(vstore);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return left->is_entailed(vstore) || right->is_entailed(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return left->is_disentailed(vstore) && right->is_disentailed(vstore);
  }

  CUDA void print(const VStore& vstore) const override {
    left->print(vstore);
    printf(" \\/ ");
    right->print(vstore);
  }

  Propagator* neg() const {
    throw new std::runtime_error("Negation of logical or constraints unimplemented.");
  }

  Propagator* to_device() const override {
    Propagator* l = left->to_device();
    Propagator* r = right->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_logical_or<<<1, 1>>>(p, uid, l, r);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

// C1 /\ C2
class LogicalAnd: public Propagator {
public:
  const Propagator* left;
  const Propagator* right;

  CUDA LogicalAnd(Propagator* left, Propagator* right):
    Propagator(-1), left(left), right(right) {}

  CUDA ~LogicalAnd() {}

  CUDA bool propagate(VStore& vstore) const override {
    bool has_changed = right->propagate(vstore);
    has_changed |= left->propagate(vstore);
    return has_changed;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return left->is_entailed(vstore) && right->is_entailed(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return left->is_disentailed(vstore) || right->is_disentailed(vstore);
  }

  CUDA void print(const VStore& vstore) const override {
    left->print(vstore);
    printf(" /\\ ");
    right->print(vstore);
  }

  Propagator* neg() const {
    return new LogicalOr(left->neg(), right->neg());
  }

  Propagator* to_device() const override {
    Propagator* l = left->to_device();
    Propagator* r = right->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_logical_and<<<1, 1>>>(p, uid, l, r);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

/// b <=> C
class ReifiedProp: public Propagator {
public:
  const Var b;
  const Propagator* rhs;
  const Propagator* not_rhs;

  CUDA ReifiedProp() = default;

  ReifiedProp(Var b, Propagator* rhs) :
    Propagator(-1), b(b), rhs(rhs), not_rhs(rhs->neg()) {}

  CUDA ReifiedProp(Var b, Propagator* rhs, Propagator* not_rhs) :
    Propagator(-1), b(b), rhs(rhs), not_rhs(not_rhs) {}

  CUDA ~ReifiedProp() {}

  CUDA bool propagate(VStore& vstore) const override {
    if (vstore.view_of(b) == 0) {
      return not_rhs->propagate(vstore);
    }
    else if (vstore.view_of(b) == 1) {
      return rhs->propagate(vstore);
    }
    else if (rhs->is_entailed(vstore)) {
      return vstore.assign(b, 1);
    }
    else if (rhs->is_disentailed(vstore)) {
      return vstore.assign(b, 0);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return
         !vstore.is_top(b)
     && ((vstore.ub(b) == 0 && rhs->is_disentailed(vstore))
      || (vstore.lb(b) == 1 && rhs->is_entailed(vstore)));
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return vstore.is_top(b)
     || (vstore.ub(b) == 0 && rhs->is_entailed(vstore))
     || (vstore.lb(b) == 1 && rhs->is_disentailed(vstore));
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    vstore.print_var(b);
    printf(" <=> (");
    rhs->print(vstore);
    printf(" )");
  }

  Propagator* neg() const {
    throw new std::runtime_error("Negation of reified constraints (b <=> C) unimplemented.");
  }

  Propagator* to_device() const override {
    Propagator* device_rhs = rhs->to_device();
    Propagator* device_not_rhs = not_rhs->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_reified_prop<<<1, 1>>>(p, uid, b, device_rhs, device_not_rhs);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

template <typename T>
static T* from_vec(std::vector<T> a) {
  T* b;
  malloc2_managed(b, a.size());
  for(int i = 0; i < a.size(); ++i) {
    b[i] = a[i];
  }
  return b;
}

template <typename T>
static T* from_ptr(const T* a, int n) {
  T* b;
  malloc2_managed(b, n);
  for(int i = 0; i < n; ++i) {
    b[i] = a[i];
  }
  return b;
}

// x1 + ... + xN <= max
template<typename T, typename U>
class Sum: public Propagator {
  const int n;
  const T* terms;
  const U::neg_type max; // x1 + ... + xN - max <= 0
public:
  Sum(std::vector<T> terms, U max):
    Propagator(-1), n(terms.size()), max(max.neg()), terms(from_vec(terms))
  {}

  CUDA Sum(int n, const T* terms, U max):
    Propagator(-1), n(n), max(max.neg()), terms(terms)
  {}

  CUDA ~Sum() {
    free2((void*)terms);
  }

  CUDA Interval sum(VStore& vstore) {
    Interval s = {max.lb(vstore), max.ub(vstore)};
    for(int i = 0; i < n; ++i) {
      s.lb += terms[i].lb(vstore);
      s.ub += terms[i].ub(vstore);
    }
    return s;
  }


    bool has_changed = vstore.update_ub(x, c - vstore.lb(y));
    has_changed |= vstore.update_ub(y, c - vstore.lb(x));

  CUDA bool propagate(VStore& vstore) const override {
    Interval s = sum(vstore);
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      int ub = s.lb - terms[i].lb(vstore);
      has_changed |= terms[i].update_ub(vstore, ub >= 0 ? -ub : ub);
    }
    int ub = s.lb - max.lb(vstore);
    has_changed |= max.update_ub(vstore, ub >= 0 ? -ub : ub);
    return has_changed;
  }

  CUDA bool one_top(const VStore& vstore) const {
    if(max.is_top(vstore)) {
      return true;
    }
    for(int i = 0; i < n; ++i) {
      if(terms[i].is_top(vars[i])) {
        return true;
      }
    }
    return false;
  }

  // From the diagram above, it is clear that once `potential <= slack` holds, it holds forever in a non-top `vstore`.
  // So even if `vstore` is modified during or between the computation of the potential or slack.
  CUDA bool is_entailed(const VStore& vstore) const override {
    return !one_top(vstore) && sum(vstore).ub <= 0;
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return one_top(vstore) || sum(vstore).lb > 0;
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    for(int i = 0; i < n; ++i) {
      terms[i].print(vstore);
      if (i != n-1) printf(" + ");
    }
    printf(" <= ");
    max.neg().print(vstore);
  }

}

// x1c1 + ... + xNcN <= max where xi = [0..1].
class PseudoBoolean: public Propagator {
  const int n;
  const Var* vars;
  const int* constants;
  const int max;
public:
  PseudoBoolean(std::vector<Var> vvars, std::vector<int> vconstants, int max):
    Propagator(-1), n(vvars.size()), max(max), vars(from_vec(vvars)),
    constants(from_vec(vconstants))
  {
    assert(vvars.size() == vconstants.size());
  }

  CUDA PseudoBoolean(int n, const Var* vars, const int* constants, int max):
    Propagator(-1), n(n), max(max), vars(vars),
    constants(constants)
  {}

  CUDA ~PseudoBoolean() {
    free2((void*)vars);
    free2((void*)constants);
  }

  // Returns the maximum amount of additional resources this constraint can use if we fix all remaining boolean variables to 1.
  // LATTICE: monotone function w.r.t. `vstore`.
  // The potential can only decrease for any evolution of non-top `vstore`.
  CUDA int potential(const VStore& vstore) const {
    int potential = 0;
    for(int i=0; i < n; ++i) {
      potential += vstore.ub(vars[i]) * constants[i];
    }
    return potential;
  }

  // Returns the amount of resources that this constraint can consume while still being satisfiable.
  // LATTICE: monotone function w.r.t. `vstore`.
  // The slack can only decrease for any evolution of non-top `vstore`.
  CUDA int slack(const VStore& vstore) const {
    int current = 0;
    for(int i=0; i < n; ++i) {
      current += vstore.lb(vars[i]) * constants[i];
    }
    return max - current;
  }

  // Consider this diagram of resources used.
  // The full window represent the number of resources used if we set all Boolean variables to 1.
  //
  //     |-------------------|_____|____|------|
  //               ^               ^        ^
  //            lb = 1            max    ub = 0
  //     |______________________________|
  //                            potential
  //                         |_____|
  //                          slack
  //
  //  Propagating assign upper bound to 0, when c_i > slack.
  CUDA bool propagate(VStore& vstore) const {
    int s = slack(vstore);
    bool has_changed = false;
    // CORRECTNESS: Even if the slack changes after its computation (or even when we are computing it), it does not hinder the correctness of the propagation.
    // The reason is that whenever `constants[i] > s` it will stay true for any slack s' since s > s' by def. of the function slack.
    for(int i=0; i < n; ++i) {
      Interval x = vstore.view_of(vars[i]);
      if (vstore.lb(vars[i]) == 0 && vstore.ub(vars[i]) == 1 && constants[i] > s) {
        has_changed |= vstore.assign(vars[i], 0);
      }
    }
    return has_changed;
  }

  CUDA bool one_top(const VStore& vstore) const {
    for(int i = 0; i < n; ++i) {
      if(vstore.is_top(vars[i])) {
        return true;
      }
    }
    return false;
  }

  // From the diagram above, it is clear that once `potential <= slack` holds, it holds forever in a non-top `vstore`.
  // So even if `vstore` is modified during or between the computation of the potential or slack.
  CUDA bool is_entailed(const VStore& vstore) const {
    return
         !one_top(vstore)
      && potential(vstore) <= max;
  }

  CUDA bool is_disentailed(const VStore& vstore) const {
    bool disentailed = one_top(vstore) || slack(vstore) < 0;
    LOG(if(disentailed) {
      printf("PseudoBoolean disentailed %d: %d < 0\n", uid, slack(vstore));
    })
    return disentailed;
  }

  CUDA void print(const VStore& vstore) const {
    printf("%d: ", uid);
    for(int i = 0; i < n; ++i) {
      printf("%d * ", constants[i]);
      vstore.print_var(vars[i]);
      if (i != n-1) printf(" + ");
    }
    printf(" <= %d", max);
  }

  Propagator* neg() const {
    throw new std::runtime_error("Negation of linear inequalities unimplemented.");
  }

  Propagator* to_device() const override {
    Propagator** p;
    malloc2_managed(p, 1);
    init_linear_ineq<<<1, 1>>>(p, uid, n, vars, constants, max);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }
};

struct Constraints {
  std::vector<Propagator*> propagators;

  size_t size() {
    return propagators.size();
  }

  // Retrieve the temporal variables (those in temporal constraints).
  // It is useful for branching.
  // The array is terminated by -1.
  Var* temporal_vars(int max) {
    bool is_temporal[max] = {false};
    for(int i=0; i < size(); ++i) {
      if(TemporalProp* p = dynamic_cast<TemporalProp*>(propagators[i])) {
        is_temporal[abs(p->x)] = true;
        is_temporal[abs(p->y)] = true;
      }
    }
    int n=0;
    for(int i=0; i < max; ++i) {
      n += is_temporal[i];
    }
    Var* vars;
    malloc2_managed(vars, (n+1));
    int j = 0;
    for(int i=0; i < max; ++i) {
      if (is_temporal[i]) {
        vars[j] = i;
        j++;
      }
    }
    vars[n] = -1;
    return vars;
  }

  void init_uids() {
    for(int i = 0; i < size(); ++i) {
      propagators[i]->uid = i;
    }
  }

  void print(const VStore& vstore) {
    for(auto p : propagators) {
      p->print(vstore);
      printf("\n");
    }
  }
};

#endif

// Copyright 2021 Pierre Talbot, Frédéric Pinel, Cem Guvel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SETVAR_HPP
#define SETVAR_HPP

#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include "bitset.cuh"

struct PartialOrder {
  static void top();
  static void bot();
};

struct PoInt: PartialOrder {
  // Dummy integer structure for lattice interval class
  int element;
  PoInt () {}
  PoInt (int element): element(element) {}
  PoInt (const PoInt& rhs): element(rhs.element) {}

  static int top () {
    return limit_min();
  }

  static int bot () {
    return limit_max();
  }

  CUDA void join (const PoInt& other) {
    element = ::max<int>(element, other.element);
  }

  CUDA void meet (const PoInt& other) {
    element = ::min<int>(element, other.element);
  }
  
  bool operator == (const PoInt& rhs) const {
    return element == rhs.element;
  }

  bool operator != (const PoInt& rhs) const {
    return element != rhs.element;
  }

  bool operator > (const PoInt& rhs) const {
    return element > rhs.element;
  }

  bool operator < (const PoInt& rhs) const {
    return rhs > *this;
  }

  bool operator <= (const PoInt& rhs) const {
    return !(*this > rhs);
  }

  bool operator >= (const PoInt& rhs) const {
    return !(*this < rhs);
  }

  void print () const {
    printf("%d\n", element);
  }
};

template <size_t N>
struct Set {
  Bitset<N> element;

  CUDA Set<N> (): element() {}

  CUDA static Set<N> bot () {
    return Bitset<N>::universe();
  }

  CUDA static Set<N> top () {
    return Bitset<N>::empty();
  }

  // Lattice Operations
  void join (const Set<N>& other) {
    element.intersection_with(other.element);
  }

  void meet (const Set<N>& other) {
    element.union_with(other.element);
  }
  // <<<

  // Binary Operations
  bool operator == (const Set<N>& rhs) const {
    return element == rhs.element;
  }

  bool operator != (const Set<N>& rhs) const {
    return element != rhs.element;
  }

  bool operator > (const Set<N>& rhs) const {
    return element > rhs.element;
  }

};

template <typename T> 
struct Dual {
  T element;

  // Constructors 
  CUDA Dual<T> (T element): element(element) {}
  // <<< 

  // Static Functions 
  CUDA static Dual<T> bot () {
    return Dual<T>(T::top());
  }

  CUDA static Dual<T> top () {
    return Dual<T>(T::bot());
  }
  // <<<

  // Lattice Operations
  CUDA void join (const Dual<T>& other) {
    element.meet(other.element);
  }

  CUDA void meet (const Dual<T>& other) {
    element.join(other.element);
  }
  // <<<

  // Binary Operations
  // Against T
  bool operator == (const T& rhs) const {
    return element == rhs;
  }

  bool operator > (const T& rhs) const {
    return element > rhs;
  }

  // Against Dual<T>
  bool operator == (const Dual<T>& rhs) const {
    return element == rhs.element;
  }

  bool operator > (const Dual<T>& rhs) const {
    return element > rhs.element;
  }
  // <<< 
};

template <typename T>
inline bool operator == (const T& left, const Dual<T>& right) {
  return right == left;
}

template <typename T>
struct Interval {
  // Variables 
  T lb;
  Dual<T> ub;
  // <<<

  // Constructors
  CUDA Interval(): lb(T::top()), ub(Dual<T>::top()) {}

  CUDA Interval(T lb, Dual<T> ub): lb(lb), ub(ub) {}

  // ASK \\ Would this work? Is it even necessary?
  // CUDA Interval(T lb, T ub): lb(lb), ub(Dual<T>(ub)) {}
  // <<< 

  // Static Functions
  // ASK \\ should this be included?
  CUDA static Interval<T> bot () {
    return { T::bot(), Dual<T>::bot() };
  }
  // <<< 

  // Lattice Operations
  CUDA Interval<T> join (const Interval<T>& other) {
    return { lb.join(other.lb), ub.join(other.ub) };
  }

  CUDA Interval<T> meet (const Interval<T>& other) {
    return { lb.meet(other.lb), ub.meet(other.ub) };
  }

  CUDA void join_with (const Interval& other) {
    lb.join_with(other.lb);
    ub.join_with(other.ub);
  }

  CUDA void meet_with (const Interval& other) {
    lb.meet_with(other.lb);
    ub.meet_with(other.ub);
  }

  // For Checking Mostly
  CUDA bool is_assigned () const {
    return lb == ub;
  }

  CUDA bool is_top () const {
    return lb > ub;
  }
  // <<< 

  //TODO negation 

  // Binary Operations
  // Against T
  CUDA bool operator == (const T& rhs) const {
    return lb == rhs && ub == rhs;
  }

  // Against Inteval<T>
  CUDA bool operator == (const Interval<T>& rhs) const {
    return lb == rhs.lb && ub == rhs.ub;
  }

  CUDA bool operator != (const Interval<T>& rhs) const {
    return lb != rhs.lb || ub != rhs.ub;
  }
  // <<< 

};

template <typename T>
CUDA bool operator < (const T& lhs, const T& rhs) {
  return rhs > lhs;
}

template <typename T>
CUDA bool operator < (const T& lhs, const Dual<T>& rhs) {
  return rhs > lhs;
}

template <typename T>
CUDA bool operator >= (const T& lhs, const T& rhs) {
  return !(lhs < rhs);
}

template <typename T>
class VStore {
  Array<Interval<T>> data;
  bool top;

  // The names don't change during solving. We want to avoid useless copies.
// Unfortunately, static member are not supported in CUDA, so we use an instance variable which is never copied.
  char** names;
  size_t names_len;
public:

  void init_names(std::vector<std::string>& vnames) {
    names_len = vnames.size();
    malloc2_managed(names, names_len);
    for(int i=0; i < names_len; ++i) {
      int len = vnames[i].size();
      malloc2_managed(names[i], len + 1);
      for(int j=0; j < len; ++j) {
        names[i][j] = vnames[i][j];
      }
      names[i][len] = '\0';
    }
  }

  void free_names() {
    for(int i = 0; i < names_len; ++i) {
      free2(names[i]);
    }
    free2(names);
  }

  template<typename Allocator>
  CUDA VStore(int nvar, Allocator& allocator):
    data(nvar, allocator), top(false)
  {}

  template<typename Allocator>
  CUDA VStore(const VStore& other, Allocator& allocator):
    data(other.data, allocator),
    top(false),
    names(other.names), names_len(other.names_len)
  {}

  VStore(int nvar): data(nvar), top(false) {}
  VStore(const VStore& other): data(other.data), top(false),
    names(other.names), names_len(other.names_len) {}

  VStore(): data(0), top(false) {}

  CUDA void reset(const VStore& other) {
    assert(size() == other.size());
    for(int i = 0; i < size(); ++i) {
      data[i] = other.data[i];
    }
    top = other.top;
  }

  CUDA bool all_assigned() const {
    for(int i = 0; i < size(); ++i) {
      if(!data[i].is_assigned()) {
        return false;
      }
    }
    return true;
  }

  CUDA bool is_top() const {
    return top;
  }

  CUDA bool is_top(Var x) const {
    return data[x].is_top();
  }

  CUDA const char* name_of(Var x) const {
    return names[x];
  }

  CUDA void print_var(Var x) const {
    printf("%s", names[x]);
  }

  CUDA void print_view(Var* vars) const {
    for(int i=0; vars[i] != -1; ++i) {
      print_var(vars[i]);
      printf(" = ");
      data[vars[i]].print();
      printf("\n");
    }
  }

  CUDA void print() const {
    // The first variable is the fake one, c.f. `ModelBuilder` constructor.
    for(int i=1; i < size(); ++i) {
      print_var(i);
      printf(" = ");
      data[i].print();
      printf("\n");
    }
  }

private:

  CUDA void update_top(Var x) {
    if(data[x].is_top()) {
      top = true;
    }
  }

public:

  // lb <= x <= ub
  CUDA void dom(Var x, Interval itv) {
    data[x] = itv;
    update_top(x);
  }

  CUDA bool update_lb(Var i, int lb) {
    if (data[i].lb < lb) {
      LOG(printf("Update LB(%s) with %d (old = %d) in %p\n", names[i], lb, data[i].lb, this));
      data[i].lb = lb;
      update_top(i);
      return true;
    }
    return false;
  }

  CUDA bool update_ub(Var i, int ub) {
    if (data[i].ub > ub) {
      LOG(printf("Update UB(%s) with %d (old = %d) in %p\n", names[i], ub, data[i].ub, this));
      data[i].ub = ub;
      update_top(i);
      return true;
    }
    return false;
  }

  CUDA bool update(Var i, Interval itv) {
    bool has_changed = update_lb(i, itv.lb);
    has_changed |= update_ub(i, itv.ub);
    return has_changed;
  }

  CUDA bool assign(Var i, int v) {
    return update(i, {v, v});
  }

  

  CUDA Interval& operator[](size_t i) {
    return data[i];
  }

  CUDA const Interval& operator[](size_t i) const {
    return data[i];
  }

  CUDA int lb(Var i) const {
    return data[i].lb;
  }

  CUDA int ub(Var i) const {
    return data[i].ub;
  }

  CUDA size_t size() const { return data.size(); }
};

#endif

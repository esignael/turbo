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


struct Int {
  // Dummy integer structure for lattice interval class
  int a;
  Int (): a(10) {}
  Int (int a): a(a) {}

  static int limit_min () {
    return -100;
  }

  static int limit_max () {
    return 99;
  }

  void join_with (const Int& other) {
    // TODO \\ change this with ::max<int> or and equiv
    a = a > other.a ? a: other.a;
  }

  void meet_with (const Int& other) {
    // TODO \\ change this with ::min<int> or and equiv
    a = a > other.a ? other.a: a;
  }

  CUDA Int join (const Int& other) const {
    // TODO \\ change this with ::max<int> or and equiv
    return { a > other.a ? other.a: a };
  }

  CUDA Int meet (const Int& other) const {
    // TODO \\ change this with ::min<int> or and equiv
    return { a > other.a ? a: other.a };
  }
  
  bool operator == (const Int& rhs) const {
    return a == rhs.a;
  }

  bool operator != (const Int& rhs) const {
    return a != rhs.a;
  }

  bool operator > (const Int& rhs) const {
    return a > rhs.a;
  }
};

template <size_t N>
struct Set {
  Bitset<N> element;

  CUDA Set<N> (): element() {}

  CUDA static Set<N> limit_max () {
    return Bitset<N>::universe();
  }

  CUDA static Set<N> limit_min () {
    return Bitset<N>::empty();
  }

  // Lattice Operations
  void join_with (const Set<N>& other) {
    element.intersection_with(other.element);
  }

  void meet_with (const Set<N>& other) {
    element.union_with(other.element);
  }

  Set<N> join (const Set<N>& other) const {
    return element.set_intersection(other.element);
  }

  Set<N> meet (const Set<N>& other) const {
    return element.set_union(other.element);
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
  CUDA static Dual<T> limit_max () {
    return Dual<T>(T::limit_min());
  }

  CUDA static Dual<T> limit_min () {
    return Dual<T>(T::limit_max());
  }
  // <<<

  // Lattice Operations
  CUDA void join_with(const Dual<T>& other) {
    element.meet_with(other.element);
  }

  CUDA void meet_with(const Dual<T>& other) {
    element.join_with(other.element);
  }

  CUDA Dual<T> join (const Dual<T>& other) {
    return { element.meet(other.element) };
  }

  CUDA Dual<T> meet (const Dual<T>& other) {
    return { element.join(other.element) };
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
  CUDA Interval(): lb(T::limit_min()), ub(Dual<T>::limit_min()) {}

  CUDA Interval(T lb, Dual<T> ub): lb(lb), ub(ub) {}

  // ASK \\ Would this work? Is it even necessary?
  // CUDA Interval(T lb, T ub): lb(lb), ub(Dual<T>(ub)) {}
  // <<< 

  // Static Functions
  // ASK \\ should this be included?
  CUDA static Interval<T> limit_max () {
    return { T::limit_max(), Dual<T>::limit_max() };
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

template <size_t array_size>
struct SetInterval{
  Bitset<array_size> lb, ub;

  CUDA SetInterval () { ub.complement(); }

  CUDA SetInterval (const Bitset<array_size> &l, const Bitset<array_size> &u): lb(l), ub(u) {}

  CUDA void inplace_join (SetInterval const &other) {
    lb = lb.set_union(other.lb);
    ub = ub.set_intersection(other.ub);
  }

  CUDA bool is_assigned () const {
    return lb == ub;
  }

  CUDA bool is_top () const {
    return lb > ub;
  }

  CUDA void complement () { 
    Bitset<array_size> temp = ub.complement();
    ub = lb.complement();
    lb = temp;
  }

  CUDA bool operator == (const SetInterval& other) const {
    return lb == other.lb && ub.other.ub;
  }

  CUDA bool operator != (const SetInterval& other) const {
    return lb != other.lb || ub != other.ub;
  }

  // Temp

  CUDA bool update_lb (const Bitset<array_size>& x) {
    if (lb < x) {
      lb = x;
      return true;
    }
    return false;
  }

  CUDA bool update_lb (int x) {
    if (lb.contains(x)) { return false; }
    lb.add(x);
    return true;
  }
};

#endif

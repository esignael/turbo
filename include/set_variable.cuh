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

#ifndef SETVAR_HPP
#define SETVAR_HPP

#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include "bitset.cuh"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::view_of`.

template <size_t array_size>
struct SetVariable {
   Bitset<array_size> lb, ub;

   CUDA SetVariable () { ub.complement(); }

   CUDA SetVariable (const Bitset<array_size> &l, const Bitset<array_size> &u): lb(l), ub(u) {}

   CUDA void inplace_join (SetVariable const &other) {
      lb = lb.set_union(other.lb);
      ub = ub.set_intersection(other.ub);
   }

   CUDA bool is_assigned () const {
      return lb.is_equiv(ub);
   }

   CUDA bool is_top () const {
      return lb.is_superset_of(ub) && lb.is_neq(ub);
   }

   CUDA void complement () { 
      Bitset<array_size> temp = ub.complement();
      ub = lb.complement();
      lb = temp;
   }

   CUDA bool is_eq (SetVariable const &other) {
      return lb.is_equiv(other.lb) && ub.is_equiv(other.ub);
   }

   CUDA bool is_neq (SetVariable const &other) {
      //TODO
      return lb.is_neq(other.lb) || ub.is_neq(other.ub);
   }
};

#endif

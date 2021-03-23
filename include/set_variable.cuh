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
      return true;
   }
};

#endif

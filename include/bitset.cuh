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

#ifndef BITSET_HPP
#define BITSET_HPP

#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include "cuda_helper.hpp"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::view_of`.

struct Bitset {
   int* set;
   size_t n;

   const int set_size = sizeof(int) * CHAR_BIT;
   const int element_size = sizeof(int) * CHAR_BIT;

   Bitset () {
      n = 10 ;
      malloc2_managed(set, n);
   }

   ~Bitset () { cudaFree(set); }

   Bitset(Bitset const &other): n(other.n) {
      malloc2_managed(set, n);
      for (int i=0; i < n; i++) {
         set[i] = other.set[i];
      }
   }

   void add (int x) {
      // if x is not in bounds, 
      int bound = element_size * n / 2;
      bound = (x >= -bound) & (x < bound);
      int bound_mask = (bound << (element_size - 1)) >> (element_size -1);
      // bound_mask will turn the assignment to set[0] |= 0, hence not changing anything

      // if x is negative,
      int sign_mask = ((x < 0) << (element_size - 1)) >> (element_size -1);
      int f = ((element_size * n) & sign_mask) + x;
      // f will give the 2's comp, depending on the size of the this->set

      int array_index = f / set_size;
      int bit_index = f % set_size;

      set[array_index & bound_mask] |= (1 << bit_index) & bound_mask;
   }

   void remove (int x) {
      int bound = element_size * n / 2;
      bound = (x >= -bound) & (x < bound);
      int bound_mask = (bound << (element_size - 1)) >> (element_size -1);
      // bound_mask will turn the assignment to set[0] |= 0, hence not changing anything

      // if x is negative,
      int sign_mask = ((x < 0) << (element_size - 1)) >> (element_size -1);
      int f = ((element_size * n) & sign_mask) + x;
      // f will give the 2's comp, depending on the size of the this->set

      int array_index = f / set_size;
      int bit_index = f % set_size;

      set[array_index & bound_mask] &= (1 << bit_index) & bound_mask;
   }


/*
   void add (short int x) {
      int j = (~(-1 << 17)) & x;
      set[j / set_size] |= (long long int)1 << (j % set_size);
   }
   */

   void remove (short int x) {
      int j = (~(-1 << 17)) & x;
      set[j / set_size] &= ~((long long int)1 << (j % set_size));
   }

   int size() const {
      int ret = 0;
      for (int i=0; i <= n; ++i){ long long unsigned inter = set[i];
         for(;inter != 0; ++ret){
            inter &= inter - 1;
         }
      }
      return ret;
   }

   Bitset diff (Bitset const &other) const {
      Bitset res;
      for (int i=0; i <= n; ++i) {
         res.set[i] = ~(other.set[i] & set[i]);
         res.set[i] &= set[i];
      }
      return res;
   }


   bool is_superset_of (Bitset const &other) const {
      long long unsigned inter = 0;
      for(int i=0;(inter == 0) && (i <= n);++i){
         inter = other.set[i] ^ set[i];
         inter &= other.set[i];
      }
      return inter == 0;
   }

   bool is_subset_of (Bitset const &other) const {
      long long unsigned inter = 0;
      for(int i=0;(inter == 0) && (i <= n);++i){
         inter = other.set[i] ^ set[i];
         inter &= set[i];
      }
      return inter == 0;
   }

   bool is_equiv (Bitset const &other) const {
      int i = 0;
      for(;(set[i] == other.set[i]) && (i <= n); ++i){}
      return set[i] == other.set[i];
   }

   bool contains (short int x) const {
      int j = (~(-1 << 17)) & x;
      return set[j / set_size] & (long long int)1 << (j % set_size);
   }

   Bitset set_union (Bitset const &x) const {
      Bitset ret;
      for(int i=0;i <= n; ++i){
         ret.set[i] = x.set[i] | set[i];
      }
      return ret;
   }

   Bitset set_intersection (Bitset const &x) const {
      Bitset ret;
      for(int i=0;i <= n; ++i){
         ret.set[i] = x.set[i] & set[i];
      }
      return ret;
   }

   void print() const {
      for (int j = 0;j < n; ++j){
         printf("%4d: ", j);
         for (int i = 0; i < set_size; i++){
            if (!(i % 8)) { printf("|"); }
            printf("%d",1 & (set[j] >> i));
         }
         printf("\n");
      }
   }
};

#endif

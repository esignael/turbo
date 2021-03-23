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

template <size_t array_size>
struct Bitset {
   int set[array_size];
   size_t n;

   const int set_size = sizeof(int) * CHAR_BIT;
   const int element_size = sizeof(int) * CHAR_BIT;

   CUDA Bitset (): n(array_size), set(){}

   //CUDA ~Bitset (){}

   // v copy constructor
   CUDA Bitset(Bitset const &other): n(other.n), set() {
      memcpy(set, other.set, n);
   }

   // v same as copy (for most part)
   CUDA Bitset& operator = (Bitset const &other) {
      if (this != &other) {
         memcpy(this, &other, sizeof(Bitset));
      }
      return *this;
   }

   CUDA void alt_add (int x) {
      int bound = element_size * n / 2;
      assert(x >= -bound && x < bound);

      int bit_index = (32 + (x % 32)) % 32;
      int row_index = (39 + ((x + 32 - bit_index) / 32)) % 40;

      set[row_index] |= 1 << bit_index;
   }

   CUDA void alt_remove (int x) {
      int bound = element_size * n / 2;
      assert(x >= -bound && x < bound);

      int bit_index = (32 + (x % 32)) % 32;
      int row_index = (39 + ((x + 32 - bit_index) / 32)) % 40;

      // Could be changed to xor if 'x' is known to be in set
      set[row_index] &= ~(1 << bit_index);
   }

   CUDA bool alt_contains (int x) {
      int bound = element_size * n / 2;
      assert(x >= -bound && x < bound);

      int bit_index = (32 + (x % 32)) % 32;
      int row_index = (39 + ((x + 32 - bit_index) / 32)) % 40;

      return set[row_index] & (1 << bit_index);
   }

   CUDA void add (int x) {
      // if x is not in bounds, 
      int bound = element_size * n / 2;
      bound = (x >= -bound) & (x < bound);
      //TODO: ^ make assersion here
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

   CUDA void remove (int x) {
      int bound = element_size * n / 2;
      bound = (x >= -bound) & (x < bound);
      int bound_mask = (bound << (element_size - 1)) >> (element_size -1);
      // bound_mask will turn the assignment to set[0] &= 1, hence not changing anything

      // if x is negative,
      int sign_mask = ((x < 0) << (element_size - 1)) >> (element_size -1);
      int f = ((element_size * n) & sign_mask) + x;
      // f will give the 2's comp, depending on the size of the this->set

      int array_index = f / set_size;
      int bit_index = f % set_size;

      set[array_index & bound_mask] &= ~(1 << bit_index) | (~bound_mask);
   }

   CUDA bool contains (int x) const {
      int bound = element_size * n / 2;
      bound = (x >= -bound) & (x < bound);
      int bound_mask = (bound << (element_size - 1)) >> (element_size -1);
      // bound_mask will turn the assignment to set[0] &= 1, hence not changing anything

      // if x is negative,
      int sign_mask = ((x < 0) << (element_size - 1)) >> (element_size -1);
      int f = ((element_size * n) & sign_mask) + x;
      // f will give the 2's comp, depending on the size of the this->set

      int array_index = f / set_size;
      int bit_index = f % set_size;
      return set[array_index & bound_mask] & ((1 << bit_index) & bound_mask);
   }

   CUDA bool is_superset_of (Bitset const &other) const {
      int inter = 0;
      for(int i=0;(inter == 0) && (i < n);++i){
         inter = other.set[i] ^ set[i];
         inter &= other.set[i];
      }
      return inter == 0;
   }

   CUDA bool is_subset_of (Bitset const &other) const {
      // TODO: make it simpler
      int inter = 0;
      for(int i=0;(inter == 0) && (i < n);++i){
         inter = other.set[i] ^ set[i];
         inter &= set[i];
      }
      return inter == 0;
   }

   CUDA bool is_equiv (Bitset const &other) const {
      int i = 0;
      for(;(set[i] == other.set[i]) && (i < n); ++i){}
      return set[i] == other.set[i];
   }

   CUDA bool is_neq (Bitset const &other) const {
      return !is_equiv(other);
      // TODO: make it simpler
      int i = 0;
      for(;(set[i] == other.set[i]) && (i < n); ++i){}
      return set[i] != other.set[i];
   }

   CUDA int size() const {
      int ret = 0;
      for (int i=0; i < n; ++i){ 
         int inter = set[i];
         for(;inter != 0; ++ret){
            inter &= inter - 1;
         }
      }
      return ret;
   }

    CUDA Bitset diff (Bitset const &other) const {
      Bitset res;
      for (int i=0; i < n; ++i) {
         res.set[i] = ~(other.set[i] & set[i]);
         res.set[i] &= set[i];
      }
      return res;
   }
   
   CUDA void cuda_difference (Bitset const &other) {
      int i = threadIdx.x + (blockIdx.x) * blockDim.x;
      int limit_mask = ((i < n) << (element_size - 1)) >> (element_size -1);
      i &= limit_mask;
      set[i] &= ~(other.set[i] & set[i] & limit_mask);
   }

   CUDA Bitset set_union (Bitset const &x) const {
      Bitset ret;
      for(int i=0;i < n; ++i){
         ret.set[i] = x.set[i] | set[i];
      }
      return ret;
   }

   CUDA Bitset set_intersection (Bitset const &x) const {
      Bitset ret;
      for(int i=0;i < n; ++i){
         ret.set[i] = x.set[i] & set[i];
      }
      return ret;
   }

   //TODO: with return value
   CUDA void complement() {
      for (int i=0; i < n; ++i) {
         set[i] = ~set[i];
      }
   }

   CUDA void print() const {
      for (int j = 0;j < n; ++j){
         printf("%4d: ", j);
         for (int i = 0; i < set_size; i++){
            if (!(i % 8)) { printf("|"); }
            printf("%d",1 & (set[j] >> i));
         }
         printf("|\n");
      }
   }

   CUDA int upper_bound() const {
      // only works for 40 (and for even numbered arrays)
      int i = 19;
      for (int j=0; j < n; ++j) {
         if (i == -1) { i = 39;}
         //if (set[i] == 0) { --i; continue; }
         for (int s=1; s <= element_size; ++s) {
            if(set[i] & (1 << (element_size - s))) {
               //printf("\nresult: %i", j);
               printf("\ns result: %i", s);
               printf("\ni result: %i", i);
               printf("\nresult: %i", element_size*(20-j)-s);
               int sol = element_size * (20-j)-s;
               printf("\nbit index: %i", (32 + (sol % 32))%32);
               printf("\nrow index: %i", (39 + ((sol + s) / 32)) % 40);
               return element_size * (20 - j) - s;
            }
         }
         --i;

      }
         printf("%i, ", i);
   }

   CUDA int interval() const {
      int ub, lb;
      for (int j=0; j < n / 2; ++j) {

      }

   }

};

#endif

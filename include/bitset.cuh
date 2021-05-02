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

template <size_t n>
struct Bitset {
  // TODO \\
  // 1) Universe and Empty statics
  // 2) Reorganize the template
  // 3) Fix the complement method
  int set[n * 2];

  const int n2 = n * 2;
  const int m = sizeof(int) * CHAR_BIT;

  CUDA Bitset (): set(){}

  //CUDA ~Bitset (){}

  // v copy constructor
  CUDA Bitset(Bitset const &other): set() {
    memcpy(set, other.set, n2);
  }

  // v same as copy (for most part)
  CUDA Bitset& operator = (Bitset const &other) {
    if (this != &other) {
      memcpy(this, &other, sizeof(Bitset));
    }
    return *this;
  }

  CUDA void add (int x) {
    // |  checking if the given parameter is within the bounds of
    // v  what the bitset can contain
    int bound = m * n;
    assert(x >= -bound && x < bound);
    // the inner modulus is used to lower the value,
    // the m addition is used to convert negative values to 
    // positive ones, since after the initial modulus, |x| < m
    // meaning x + m > 0. Taking the modulus a second time makes
    // sure that if x > 0 then x < m.
    int bit_index = (m + (x % m)) % m;
    // this is much harder to explain `:D
    int row_index = (n2 - 1 + ((x + m - bit_index) / m)) % n2;

    set[row_index] |= 1 << bit_index;
  }

  CUDA void remove (int x) {
    int bound = m * n;
    assert(x >= -bound && x < bound);

    int bit_index = (m + (x % m)) % m;
    int row_index = (n2 - 1 + ((x + m - bit_index) / m)) % n2;

    // Could be changed to xor if 'x' is known to be in set
    set[row_index] &= ~(1 << bit_index);
  }

  CUDA bool contains (int x) const {
    int bound = m * n;
    if (x >= -bound && x < bound) { return false; }

    int bit_index = (m + (x % m)) % m;
    int row_index = (n2 - 1 + ((x + m - bit_index) / m)) % n2;

    return set[row_index] & (1 << bit_index);
  }

  CUDA bool operator == (Bitset const &other) const {
    for (int i=0; i<n2;++i) {
      if (set[i] != other.set[i]) { return false; }
    } return true;
  }

  CUDA bool operator != (Bitset const &other) const {
    return !(*this == other);
  }

  CUDA bool operator >= (Bitset const &other) const {
    bool temp = true;
    int inter = 0;
    for (int i=0;i<n2;++i) {
      inter = other.set[i] ^ set[i];
      inter &= other.set[i];
      temp &= (inter == 0);
    }
    return temp;
  }

  CUDA bool operator < (Bitset const &other) const {
    return !(*this >= other);
  }

  CUDA bool operator <= (Bitset const &other) const {
    return other >= *this;
  }

  CUDA bool operator > (Bitset const &other) const {
    return other < *this;
  }

  CUDA int size() const {
    int ret = 0;
    for (int i=0; i < n2; ++i){ 
      int inter = set[i];
      for(;inter != 0; ++ret){
        inter &= inter - 1;
      }
    }
    return ret;
  }

  CUDA Bitset diff (Bitset const &other) const {
    Bitset res;
    for (int i=0; i < n2; ++i) {
      res.set[i] = ~(other.set[i] & set[i]);
      res.set[i] &= set[i];
    }
    return res;
  }

  CUDA Bitset set_union (Bitset const &x) const {
    Bitset ret;
    for(int i=0;i < n; ++i){
      ret.set[i] = x.set[i] | set[i];
    }
    return ret;
  }

  CUDA void union_with (const Bitset<n>& other) {
    for (int i = 0; i < n; ++i) {
      set[i] |= other.set[i];
    }
  }

  CUDA Bitset set_intersection (Bitset const &x) const {
    Bitset ret;
    for(int i=0;i < n; ++i){
      ret.set[i] = x.set[i] & set[i];
    }
    return ret;
  }

  CUDA void intersection_with (const Bitset<n>& other) {
    for (int i = 0; i < n; ++i) {
      set[i] &= other.set[i];
    }
  }

  //TODO: with return value
  CUDA void complement() {
    for (int i=0; i < n2; ++i) {
      set[i] = ~set[i];
    }
  }

  CUDA void print() const {
    for (int j = 0;j < n2; ++j){
      printf("%4d: ", j);
      for (int i = 0; i < m; i++){
        if (!(i % 8)) { printf("|"); }
        printf("%d",1 & (set[j] >> i));
      }
      printf("|\n");
    }
  }

  CUDA int max() const {
    int rel_i = n - 1;
    for (int j=0; j < n2; ++j) {
      if (rel_i == -1) { rel_i = n2-1;}
      // Maybe for CPU
      //if (set[rel_i] == 0) { --rel_i; continue; }
      for (int s=1; s <= m; ++s) {
        if(set[rel_i] & (1 << (m - s))) {
          return m * (n - j) - s;
        }
      }
      --rel_i;
    }
    return -n * m - 1;
  }

  CUDA int min() const {
    int rel_i = n;
    for (int j=0; j < n2 ;++j) {
      if (rel_i == n2) { rel_i = 0; }
      for (int bit_i=0; bit_i < m;++bit_i) {
        if (set[rel_i] & (1 << bit_i)) {
          return m * (j - n) + bit_i;
        }
      }
      ++rel_i;
    }
    return n * m;
  }

  CUDA int interval() const {
    int ub, lb;
    return 0;
  }

};

#endif

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

#ifndef SSTORE_HPP
#define SSTORE_HPP

#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include "set_variable.cuh"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::view_of`.
typedef int Var;
struct no_copy_tag {};
struct device_tag {};

template <size_t array_size, size_t variable_size>
struct SetStore {
  SetVariable<variable_size>* data;
  size_t n;

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

  SetStore() {
    n = array_size;
    malloc2_managed(data, n);
  }
  SetStore(const SetStore& other) = delete;

  SetStore(const SetStore& other, no_copy_tag) {
    n = other.n;
    names = other.names;
    names_len = other.names_len;
    malloc2_managed(data, n);
  }

  CUDA SetStore(const SetStore& other, no_copy_tag, device_tag) {
    n = other.n;
    names = other.names;
    names_len = other.names_len;
    malloc2(data, n);
  }

  CUDA void reset(const SetStore& other) {
    assert(n == other.n);
    for(int i = 0; i < n; ++i) {
      data[i] = other.data[i];
    }
  }

  ~SetStore() {
    cudaFree(data);
  }

  CUDA bool all_assigned() const {
    for(int i = 0; i < n; ++i) {
      if(!data[i].is_assigned()) {
        return false;
      }
    }
    return true;
  }

  CUDA bool is_top() const {
    for(int i = 0; i < n; ++i) {
      if(data[i].is_top()) {
        return true;
      }
    }
    return false;
  }

  CUDA bool is_top(Var x) const {
    return data[abs(x)].is_top();
  }

  CUDA const char* name_of(Var x) const {
    return names[abs(x)];
  }

  CUDA void print_var(Var x) const {
    printf("%s%s", (x < 0 ? "-" : ""), names[abs(x)]);
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
    for(int i=1; i < n; ++i) {
      print_var(i);
      printf(" = ");
      data[i].print();
      printf("\n");
    }
  }

  // lb <= x <= ub
  CUDA void dom(Var x, SetVariable<variable_size> itv) {
    data[x] = itv;
  }

  bool update_lb (Var i, Bitset<variable_size> lb) {
     int sign = ((i < 0) << (sizeof(int) - 1) >> 

  CUDA bool update_lb(Var i, Bitset<variable_size> lb) {
    if(i >= 0) {
      if (data[i].lb.is_superset_of(lb)) {
        LOG(printf("Update LB(%s) with %d (old = %d) in %p\n", names[i], lb, data[i].lb, this));
        data[i].lb = lb;
        return true;
      }
    }
    else {
      if (data[-i].ub > -lb) {
        LOG(printf("Update UB(%s) with %d (old = %d) in %p\n", names[-i], -lb, data[-i].ub, this));
        data[-i].ub = -lb;
        return true;
      }
    }
    return false;
  }

  CUDA bool update_ub(Var i, int ub) {
    if(i >= 0) {
      if (data[i].ub > ub) {
        LOG(printf("Update UB(%s) with %d (old = %d) in %p\n", names[i], ub, data[i].ub, this));
        data[i].ub = ub;
        return true;
      }
    }
    else {
      if (data[-i].lb < -ub) {
        LOG(printf("Update LB(%s) with %d (old = %d) in %p\n", names[-i], -ub, data[-i].lb, this));
        data[-i].lb = -ub;
        return true;
      }
    }
    return false;
  }

  CUDA bool update(Var i, SetVariable<variable_size> itv) {
    bool has_changed = update_lb(i, itv.lb);
    has_changed |= update_ub(i, itv.ub);
    return has_changed;
  }

  CUDA bool assign(Var i, int v) {
    return update(i, {v, v});
  }

  CUDA SetVariable<variable_size> view_of(Var i) const {
    return i < 0 ? data[-i].neg() : data[i];
  }

  CUDA int lb(Var i) const {
    return view_of(i).lb;
  }

  CUDA int ub(Var i) const {
    return view_of(i).ub;
  }

  CUDA size_t size() const { return n; }
};

#endif

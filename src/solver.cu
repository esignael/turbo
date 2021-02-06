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

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <new>
#include <chrono>

#include "solver.cuh"
#include "vstore.cuh"
#include "constraints.cuh"
#include "cuda_helper.hpp"
#include "statistics.cuh"
#include "status.cuh"
#include "search.cuh"

#ifdef SEQUENTIAL

template <typename T>
bool propagate(std::vector<T>& constraints, VStore& vstore, PropagatorsStatus& pstatus) {
  bool has_changed = false;
  for(auto p : constraints) {
    bool has_changed2 = p.propagate(vstore);
    has_changed |= has_changed2;
    Status s = has_changed2 ? UNKNOWN : IDLE;
    if(p.is_entailed(vstore)) {
      s = ENTAILED;
    }
    if(p.is_disentailed(vstore)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
  return has_changed;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  INFO(constraints.print(*vstore));
  Statistics stats;
  VStore best_sol = VStore(vstore->size());
  Var* temporal_vars = constraints.temporal_vars(vstore->size());
  SharedData shared_data = SharedData(vstore, constraints.size());

  auto t1 = std::chrono::high_resolution_clock::now();

  shared_data.into_device_mem();
  Stack stack(*(shared_data.vstore));
  Interval best_bound = {limit_min(), limit_max()};
  INFO(printf("starting search with %p\n", shared_data.vstore));

  while(shared_data.exploring) {
    // I. Propagation
    VStore& vstore = *(shared_data.vstore);
    PropagatorsStatus& pstatus = *(shared_data.pstatus);
    bool has_changed = true;
    while(has_changed && pstatus.join() < ENTAILED) {
      has_changed = propagate(constraints.temporal, vstore, pstatus);
      has_changed |= propagate(constraints.reifiedLogicalAnd, vstore, pstatus);
      has_changed |= propagate(constraints.linearIneq, vstore, pstatus);
    }
    // We propagate once more to verify that all propagators are really entailed.
    if(pstatus.join() == ENTAILED) {
      propagate(constraints.temporal, vstore, pstatus);
      propagate(constraints.reifiedLogicalAnd, vstore, pstatus);
      propagate(constraints.linearIneq, vstore, pstatus);
    }
    // II. Branching
    one_step(stack, best_bound, shared_data.pstatus->join(),
      &shared_data, &stats, &best_sol, minimize_x, temporal_vars);
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  stats.print();
  std::cout << "solveTime=" << duration << std::endl;
}

#else

const int PROPS_TYPE = 3;

CUDA_GLOBAL void status_k(SharedData* shared_data, int* fixpoint /* out */) {
  if(shared_data->pstatus->join() != UNKNOWN) {
    *fixpoint += 1;
  }
  else {
    *fixpoint = 0;
  }
  // INFO(printf("status_k: status->join=%d\n", *fixpoint));
  LOG(shared_data->vstore->print());
}

template<typename T>
CUDA_GLOBAL void propagate_k(SharedData* shared_data, T* props) {
  int cid = threadIdx.x + blockDim.x * blockIdx.x;
  T& p = props[cid];
  PropagatorsStatus& pstatus = *(shared_data->pstatus);
  VStore& vstore = *(shared_data->vstore);
  Status s;
  for (int i=0; i<PITER; ++i) {
    s = p.propagate(vstore) ? UNKNOWN : IDLE;
    if(p.is_entailed(vstore)) {
      s = ENTAILED;
    }
    if(p.is_disentailed(vstore)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
}

template<typename T>
T* cons_alloc(std::vector<T> &c)
{
  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    new(props + i) T(c[i]);
  }
  return props;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  INFO(constraints.print(*vstore));

  void* stats_raw;
  CUDIE(cudaMallocManaged(&stats_raw, sizeof(Statistics)));
  Statistics* stats = new(stats_raw) Statistics();

  void* best_sol_raw;
  CUDIE(cudaMallocManaged(&best_sol_raw, sizeof(VStore)));
  VStore* best_sol = new(best_sol_raw) VStore(vstore->size());

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE];
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamCreate(&streams[i]));
  }

  void *raw_interval;
  CUDIE(cudaMallocManaged(&raw_interval, sizeof(Interval)));
  Interval *best_bound = new(raw_interval) Interval();

  auto t1 = std::chrono::high_resolution_clock::now();

  void *raw_shared_data;
  CUDIE(cudaMallocManaged(&raw_shared_data, sizeof(SharedData)));
  SharedData* shared_data = new(raw_shared_data) SharedData(vstore, constraints.size());

  void *raw_stack;
  CUDIE(cudaMallocManaged(&raw_stack, sizeof(Stack)));
  Stack *stack = new(raw_stack) Stack(*(shared_data->vstore));

  auto tem_p = cons_alloc<TemporalProp>(constraints.temporal);
  auto rei_p = cons_alloc<ReifiedLogicalAnd>(constraints.reifiedLogicalAnd);
  auto lin_p = cons_alloc<LinearIneq>(constraints.linearIneq);

  int *fixpoint;
  CUDIE(cudaMallocManaged(&fixpoint, sizeof(*fixpoint)));
  *fixpoint = 0;

  while (shared_data->exploring) {
    do {
      propagate_k<TemporalProp><<<constraints.temporal.size(), 1, 0, streams[0]>>>(shared_data, tem_p);
      propagate_k<ReifiedLogicalAnd><<<constraints.reifiedLogicalAnd.size(), 1, 0, streams[1]>>>(shared_data, rei_p);
      propagate_k<LinearIneq><<<constraints.linearIneq.size(), 1, 0, streams[2]>>>(shared_data, lin_p);
      CUDIE(cudaDeviceSynchronize());
      status_k<<<1,1>>>(shared_data, fixpoint);
      CUDIE(cudaDeviceSynchronize());
    } while (*fixpoint < constraints.size());
    *fixpoint = 0;
    search<<<1, 1>>>(stack, shared_data, stats, best_sol, minimize_x, temporal_vars, best_bound);
    CUDIE(cudaDeviceSynchronize());
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  CUDIE(cudaFree(fixpoint));

  CUDIE(cudaFree(tem_p));
  CUDIE(cudaFree(rei_p));
  CUDIE(cudaFree(lin_p));

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  stats->print();
  std::cout << "solveTime=" << duration << std::endl;

  best_sol->~VStore();
  CUDIE(cudaFree(best_sol_raw));

  CUDIE(cudaFree(temporal_vars));

  shared_data->~SharedData();
  CUDIE(cudaFree(raw_shared_data));

  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamDestroy(streams[i]));
  }
  CUDIE(cudaFree(raw_stack));
  CUDIE(cudaFree(raw_interval));
}

#endif

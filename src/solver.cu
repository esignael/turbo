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
#include <cstdio>
#include <chrono>
#include <thread>

#include "solver.cuh"
#include "vstore.cuh"
#include "propagators.cuh"
#include "cuda_helper.hpp"
#include "statistics.cuh"
#include "search.cuh"

__device__ int decomposition = 0;

// #define SHMEM_SIZE 65536
#define SHMEM_SIZE 44000

#define IN_GLOBAL_MEMORY

CUDA_GLOBAL void search_k(
    Array<Pointer<TreeAndPar>>* trees,
    VStore* root,
    Array<Pointer<Propagator>>* props,
    Array<Var>* branching_vars,
    Pointer<Interval>* best_bound,
    Array<VStore>* best_sols,
    Var minimize_x,
    Array<Statistics>* blocks_stats,
    int subproblems_power,
    bool* stop)
{
  #ifndef IN_GLOBAL_MEMORY
    extern __shared__ int shmem[];
    const int n = SHMEM_SIZE;
  #endif
  int tid = threadIdx.x;
  int nodeid = blockIdx.x;
  int stride = blockDim.x;
  __shared__ int curr_decomposition;
  __shared__ int decomposition_size;
  int subproblems = pow(2, subproblems_power);

  if (tid == 0) {
    decomposition_size = subproblems_power;
    INFO(printf("decomposition = %d, %d\n", decomposition_size, subproblems));
    #ifdef IN_GLOBAL_MEMORY
      GlobalAllocator allocator;
    #else
      SharedAllocator allocator(shmem, n);
    #endif
    (*trees)[nodeid].reset(new(allocator) TreeAndPar(
      *root, *props, *branching_vars, **best_bound, minimize_x, allocator));
    curr_decomposition = atomicAdd(&decomposition, 1);
  }
  __syncthreads();
  while(curr_decomposition < subproblems && !(*stop)) {
    INFO(if(tid == 0) printf("Block %d with decomposition %d.\n", nodeid, curr_decomposition));
    (*trees)[nodeid]->search(tid, stride, *root, curr_decomposition, decomposition_size, *stop);
    if (tid == 0) {
      Statistics latest = (*trees)[nodeid]->statistics();
      if(latest.best_bound != -1 && latest.best_bound < (*blocks_stats)[nodeid].best_bound) {
        (*best_sols)[nodeid].reset((*trees)[nodeid]->best());
      }
      (*blocks_stats)[nodeid].join(latest);
      curr_decomposition = atomicAdd(&decomposition, 1);
    }
    __syncthreads();
  }
  INFO(if(tid == 0) printf("Block %d quits %d.\n", nodeid, (*blocks_stats)[nodeid].best_bound));
  // if(tid == 0)
   // printf("%d: Block %d quits %d.\n", tid, nodeid, (*blocks_stats)[nodeid].best_bound);
}

// Inspired by https://stackoverflow.com/questions/39513830/launch-cuda-kernel-with-a-timeout/39514902
// Timeout expected in seconds.
void guard_timeout(int timeout, bool& stop) {
  int progressed = 0;
  while (!stop) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    progressed += 1;
    if (progressed >= timeout) {
      stop = true;
    }
  }
}

void update_heap_limit() {
  size_t heap_limit;
  cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_limit*10);
  INFO(std::cout << "heap limit = " << heap_limit << std::endl);
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, Configuration config)
{
  // INFO(constraints.print(*vstore));
  update_heap_limit();

  Array<Var>* branching_vars = constraints.branching_vars();

  LOG(std::cout << "Start transfering propagator to device memory." << std::endl);
  auto t1 = std::chrono::high_resolution_clock::now();
  Array<Pointer<Propagator>>* props = new(managed_allocator) Array<Pointer<Propagator>>(constraints.size());
  LOG(std::cout << "props created " << props->size() << std::endl);
  for (auto p : constraints.propagators) {
    LOG(p->print(*vstore));
    LOG(std::cout << std::endl);
    (*props)[p->uid].reset(p->to_device());
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  LOG(std::cout << "Finish transfering propagators to device memory (" << duration << " ms)" << std::endl);

  t1 = std::chrono::high_resolution_clock::now();

  Array<Pointer<TreeAndPar>>* trees = new(managed_allocator) Array<Pointer<TreeAndPar>>(config.or_nodes);
  Pointer<Interval>* best_bound = new(managed_allocator) Pointer<Interval>(Interval());
  Array<VStore>* best_sols = new(managed_allocator) Array<VStore>(*vstore, config.or_nodes);
  Array<Statistics>* blocks_stats = new(managed_allocator) Array<Statistics>(config.or_nodes);

  bool* stop = new(managed_allocator) bool(false);
  // cudaFuncSetAttribute(search_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SIZE);
  int and_nodes = min((int)props->size(), config.and_nodes);
  search_k<<<config.or_nodes, and_nodes
    #ifndef IN_GLOBAL_MEMORY
      , SHMEM_SIZE
    #endif
  >>>(trees, vstore, props, branching_vars, best_bound, best_sols, minimize_x, blocks_stats, config.subproblems_power, stop);

  std::thread timeout_thread(guard_timeout, config.timeout, std::ref(*stop));
  CUDIE(cudaDeviceSynchronize());
  *stop = true;

  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  timeout_thread.join();

  Statistics statistics;
  for(int i = 0; i < blocks_stats->size(); ++i) {
    statistics.join((*blocks_stats)[i]);
  }
  GlobalStatistics gstats(vstore->size(), constraints.size(), duration, statistics);
  gstats.print();

  operator delete(best_bound, managed_allocator);
  operator delete(props, managed_allocator);
  operator delete(trees, managed_allocator);
  operator delete(branching_vars, managed_allocator);
  operator delete(best_bound, managed_allocator);
  operator delete(best_sols, managed_allocator);
}

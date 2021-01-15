#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <new>

#define CUDA __device__ __host__

#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())

__device__ __managed__ uint Act_cnt = 0;

template<typename T>CUDA T min(T a, T b) { return a<=b ? a : b; }
template<typename T>CUDA T max(T a, T b) { return a>=b ? a : b; }

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::operator[]`.
typedef size_t Var;

struct Interval {
  int lb;
  int ub;

  CUDA Interval join(Interval b) {
    lb = max<int>(lb, b.lb);
    ub = min<int>(ub, b.ub);
    return *this;
  }

  CUDA Interval neg() {
    return {-ub, -lb};
  }

  CUDA bool operator==(int x) {
    return lb == x && ub == x;
  }
};

struct VStore {
  Interval* data;
  size_t size;

  VStore(int nvar) {
    size = nvar;
    CUDIE(cudaMallocManaged(&data, sizeof(*data) * nvar));
  }

  void free() {
    CUDIE(cudaFree(data));
    size = 0;
  }

  CUDA void print_store() {
    for(int i=0; i < size; ++i) {
      printf("%d = [%d..%d]\n", i, data[i].lb, data[i].ub);
    }
  }

  // lb <= x <= ub
  CUDA void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  CUDA void update(int i, Interval itv) {
    if (i<0) {
      data[-i].lb = -itv.ub;
      data[-i].ub = -itv.lb;
    } else {
      data[i] = itv;
    }
  }

  CUDA Interval operator[](int i) {
    return i < 0 ? data[-i].neg() : data[i];
  }
};

/// x + y <= c
struct XplusYleqC {
  Var x;
  Var y;
  int c;

  CUDA XplusYleqC(Var x, Var y, int c) : x(x), y(y), c(c) {}

  CUDA void propagate(VStore& vstore)
  {
    Act_cnt++;  // abstract?
    vstore.update(x,
        vstore[x].join({vstore[x].lb, c - vstore[y].lb}));
    vstore.update(y,
        vstore[y].join({vstore[y].lb, c - vstore[x].lb}));
  }

  CUDA bool is_entailed(VStore& vstore) {
    return vstore[x].ub + vstore[y].ub <= c;
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return vstore[x].lb + vstore[y].lb > c;
  }

  CUDA XplusYleqC neg() {
    return XplusYleqC(-x, -y, -c - 1);
  }
};

/// b <=> left /\ right
struct ReifiedLogicalAnd {
  Var b;
  XplusYleqC left;
  XplusYleqC right;

  CUDA ReifiedLogicalAnd(Var b, XplusYleqC left, XplusYleqC right) :
    b(b), left(left), right(right) {}

  CUDA void propagate(VStore& vstore) {
    if (vstore[b] == 0) {
      left.neg().propagate(vstore);
      right.neg().propagate(vstore);
    }
    else if (vstore[b] == 1) {
      left.propagate(vstore);
      right.propagate(vstore);
    }
    else if (left.is_entailed(vstore) && right.is_entailed(vstore)) {
      vstore.update(b, {1, 1});
    }
    else if (left.is_disentailed(vstore) || right.is_disentailed(vstore)) {
      vstore.update(b, {0, 0});
    }
  }
};

template<typename Constraint>
__global__ void propagate_k(Constraint c, VStore* vstore) {
  c.propagate(*vstore);
}

// C1 \/ C2
struct LogicalOr {
  XplusYleqC left;
  XplusYleqC right;

  LogicalOr(XplusYleqC left, XplusYleqC right):
    left(left), right(right) {}

  CUDA void propagate(VStore& vstore) {
    if (left.is_disentailed(vstore)) {
      right.propagate(vstore);
    }
    else if (right.is_disentailed(vstore)) {
      left.propagate(vstore);
    }
  }

  CUDA bool is_entailed(VStore& vstore) {
    return left.is_entailed(vstore) || right.is_entailed(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return left.is_disentailed(vstore) && right.is_disentailed(vstore);
  }
};

// x1c1 + ... + xNcN <= max
struct LinearIneq {
  int n;
  Var* vars;
  int* constants;
  int max;

  LinearIneq(int n, Var* vars, int* constants, int max) :
    n(n), vars(vars), constants(constants), max(max) {}

  CUDA int leftover(VStore& vstore) {
    int leftover = 0;
    for(int i=0; i < n; ++i) {
      if (vstore[vars[i]].lb == 0 && vstore[vars[i]].ub == 1) {
        leftover += constants[i];
      }
    }
    return leftover;
  }

  CUDA int slack(VStore& vstore) {
    int current = 0;
    for(int i=0; i < n; ++i) {
      current += vstore[vars[i]].lb * constants[i];
    }
    return max - current;
  }

  CUDA void propagate(VStore& vstore) {
    int s = slack(vstore);
    for(int i=0; i < n; ++i) {
      Interval x = vstore[vars[i]];
      if (x.lb == 0 && x.ub == 1 && constants[i] > s) {
        vstore.update(vars[i], {0,0});
      }
    }
  }

  CUDA bool is_entailed(VStore& vstore) {
    return leftover(vstore) <= slack(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return slack(vstore) < 0;
  }
};

__global__ 
void is_active_k() {
  __shared__ uint prev;  // shared: avoids spillage to global (tbc)
  __shared__ uint curr;
  __shared__ bool inactive;
  inactive = false;
  while (1) {
    asm("nanosleep.u32 1000000000;");
    prev = curr;
    curr = Act_cnt;
    if (!inactive && curr == prev) {
      inactive = true;
      printf("no activity\n");
      break;  // for now
    } else {
      printf("active!\n");
    }
  }
}

int main() {

  // I. Declare the variable's domains.
  int nvar = 4;
  int x = 0;
  int y = 1;
  int z = 2;
  int b = 3;

  void* v;
  CUDIE(cudaMallocManaged(&v, sizeof(VStore)));
  VStore* vstore = new(v) VStore(nvar);

  vstore->dom(x, {0, 2});
  vstore->dom(y, {1, 3});
  vstore->dom(z, {2, 4});
  vstore->dom(b, {0,1});

  vstore->print_store();

  // II. Declare the constraints
  XplusYleqC c1(x,y,2);
  XplusYleqC c2(y,z,2);
  ReifiedLogicalAnd c3(b, c1, c2);

  // III. Solve the problem.
  //c3.propagate(*vstore);
  // concurrent execution of monitoring and solving using streams:
  cudaStream_t monitor, solve;
  CUDIE(cudaStreamCreate(&monitor));
  CUDIE(cudaStreamCreate(&solve));

  is_active_k<<<1,1,0,monitor>>>();
  propagate_k<ReifiedLogicalAnd><<<1,1,0,solve>>>(c3, vstore);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());
  
  vstore->print_store();

  CUDIE(cudaStreamDestroy(monitor));
  CUDIE(cudaStreamDestroy(solve));
  vstore->free();
  CUDIE(cudaFree(v));

  return 0;
}

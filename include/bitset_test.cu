#include "bitset.cuh"
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

template <size_t array_size>
__global__ void kern (Bitset<array_size> &a) {
   a.add(1);
   a.add(-1);
   a.add(-2);
   a.add(0);
   a.add(-160);
   a.add(160);
   a.add(-394);
   a.remove(-1);
   a.remove(-394);
}

template <size_t array_size>
__global__ void diff_test (Bitset<array_size> &a, Bitset<array_size> &b) {
   a.cuda_difference(b);
}

int main () {
   Bitset<40> *a = new Bitset<40>;
   Bitset<40> *b = new Bitset<40>;
   for(int i=0;i<400;++i){a->add(i); }
   for(int i=-100;i<0;++i){b->add(i); }
   /*
   a.add(1);
   a.add(-1);
   a.add(-2);
   a.add(0);
   a.add(-160);
   a.print();
   a.add(160);
   a.add(-394);
   a.remove(-1);
   a.remove(-394);
   */

   cudaDeviceSynchronize();
   kern<<<1,20>>>(*a);
   cudaDeviceSynchronize();
   b->print();

   cudaDeviceSynchronize();
   diff_test<<<1,42>>>(*b, *a);
   cudaDeviceSynchronize();
   b->print();
   a->print();
   printf("=========== B / A============");
   (b->diff(*a)).print();
   Bitset<40> c;
   c = b->diff(*a);
   printf("===========   C  ============");
   c.print();
   
   return 0;
};

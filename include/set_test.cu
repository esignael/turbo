#include "set_variable.cuh"

int main () {
   SetVariable<40> a;
   a.lb.print();
   a.ub.print();
   return 0;
}

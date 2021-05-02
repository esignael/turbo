#include "set_variable.cuh"

int main () {
  Int a;
  Int c = 7;
  std::cout << a.a << std::endl;
  a.join_with(c);
  std::cout << a.a << std::endl;
  Dual<Int> b(a);
  Dual<Int> d(c);
  std::cout << (b.meet(d)).element.a << std::endl;
  std::cout << "here" << b.element.a << std::endl;
  std::cout << (a == b) << std::endl;
  std::cout << (b == a) << std::endl;
  std::cout << (b > a) << std::endl;


  return 0;
}

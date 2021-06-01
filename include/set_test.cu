#include "set_variable.cuh"

int main () {
  std::cout << "==== PoInt Test ====" << std::endl;
  std::cout << "---- Statics ----" << std::endl;
  std::cout << "Top: " << PoInt::top() << "\nBottom: " << PoInt::bot() << std::endl;
  std::cout << "---- Constructors ----" << std::endl;
  PoInt a;
  a = 2;
  printf("PoInt a: "); a.print();
  PoInt b = 10;
  printf("PoInt b: "); b.print();
  PoInt c = a;
  a = 8;
  printf("PoInt a: "); a.print();
  printf("PoInt c: "); c.print();

  printf("---- Interval Operations ----\n");
  a.join(c);
  printf("a join c: "); a.print();
  b.meet(a);
  printf("b meet a: "); b.print();


  return 0;
}

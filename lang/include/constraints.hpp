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

#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

class Constraint {
  virtual std::vector<GuardedCommands> tell() const = 0;
  virtual std::unique_ptr<Ask> ask() const = 0;
  virtual std::unique_ptr<Constraint> neg() const = 0;
}

// x + y <= c
class TemporalConstraint: public Constraint {
  // bool positive_x;
  std::string x;
  // bool negative_x;
  std::string y;
  int c;

  std::vector<GuardedCommands> tell() const {
    auto  = ;
    auto c = ;
    auto c1 = make_true_gc(std::make_unique<TellInterval>(var,
      std::make_unique<LB>(x),
      std::make_unique<TellAnd>("-",
        std::make_unique<IntConstant>(c),
        std::make_unique<LB>(y)));)
  }
}

#endif

// I. x + y <= c

//  ==> (tell)

// true => x <- [lb(x)..c - lb(y)]
// true => y <- [lb(y)..c - lb(x)]

//  ==> (not)

// -x - y <= -c

//  ==> (ask)

// ask(x) /\ ask(y) /\ ub(x) + ub(y) <= c


// II. c1 \/ c2

//   ==> (tell)

// ask(not(c1)) => c2
// ask(not(c2)) => c1

//   ==> (not)

// not(c1) /\ not(c2)

//   ==> (ask)

// ask(c1)
// ask(c2)

// III. c1 /\ c2

//    ==> (tell)
//  true => c1
//  true => c2

//

// IV. b <=> c1

//   ==> (tell)

// lb(b) = 1 /\ ub(b) = 1 => c1
// lb(b) = 0 /\ ub(b) = 0 => not(c1)
// ask(c1) => b = 1
// ask(not(c1)) => b = 0

//   ==> (not)

// not(b) <=> c1

//   ==> (ask)

// b = 1 /\ ask(c1)
// b = 0 /\ ask(not(c1))

// V. x1c1 + x2c2 + ... + xNcN <= c

//   ==> (tell)

// true => potential <- ub(x1) * c1 + ... + ub(xN) * cN
// true => slack <- c - (lb(x1) * c1 + lb(x2) * c2 + ... + lb(xN) * cN)
// lb(x1) != ub(x1) /\ c1 > slack => x1 <- [0..0]
// ...
// lb(xN) != ub(xN) /\ cN > slack => xN <- [0..0]
// slack < 0 => false

//   ==> (not)

// x1c1 + x2c2 + ... + xNcN > c

//   ==> (ask)

// ask(x1) /\ ... /\ ask(xN) /\ potential <= max

// VI. Variable x

//   ==> (ask)

// lb(x) <= ub(x)


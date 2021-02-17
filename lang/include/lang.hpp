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

#ifndef TURBO_LANG_HPP
#define TURBO_LANG_HPP

class Expr {
  virtual std::string generate() const = 0;
  virtual ~Expr();
};

class IntConstant: public Expr {
  int v;
public:
  IntConstant(int v): v(v) {}
  ~IntConstant(){}
  std::string generate() const {
    return std::to_string(v);
  }
};

class BoolConstant: public Expr {
  bool v;
public:
  BoolConstant(int v): v(v) {}
  ~BoolConstant(){}
  std::string generate() const {
    if(v) { return "true"; }
    else { return "false"; }
  }
}

class LB : public Expr {
  std::string var
public:
  LB(std::string var): var(var) {}
  ~LB(){}
  std::string generate() const {
    return "vstore.lb(" + var + ")";
  }
};

class UB : public Expr {
  std::string var
public:
  UB(std::string var): var(var) {}
  ~UB(){}
  std::string generate() const {
    return "vstore.ub(" + var + ")";
  }
};

class BinOp: public Expr {
  std::vector<std::unique_ptr<Expr>> exprs;
  char op;
public:
  BinOp(std::vector<std::unique_ptr<Expr>> exprs, char op):
    exprs(std::move(exprs)), op(op) {}
  ~BinOp(){}
  std::string generate() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->generate() + " " + op + " ";
    }
    res.pop_back();
    res.pop_back();
    res += ")";
    return std::move(res);
  }
};

class CmpOp: public Expr {
  std::unique_ptr<Expr> left;
  std::unique_ptr<Expr> right;
  char op;
public:
  CmpOp(std::unique_ptr<Expr> left, std::unique_ptr<Expr> right, char op):
    left(std::move(left)), right(std::move(right)), op(op) {}
  ~CmpOp(){}
  std::string generate() const {
    return left->generate() + " " + op + " " + right->generate();
  }
};

// ASK language: expr <op> expr / A1 /\ A2 / true
class Ask {
  virtual std::string generate() const = 0;
  virtual ~Ask();
};

class AskBool: public Ask {
  std::unique_ptr<BoolConstant> v;
public:
  AskBool(std::unique_ptr<BoolConstant> v): v(std::move(v)) {}
  ~AskBool(){}
  std::string generate() const {
    return v->generate();
  }
};

class AskCmp: public Ask {
  std::unique_ptr<CmpOp> expr;
public:
  AskCmp(std::unique_ptr<CmpOp> expr): expr(std::movee(expr)) {}
  ~AskCmp(){}
  std::string generate() const {
    return expr->generate();
  }
};

class AskAnd: public Ask {
  std::vector<std::unique_ptr<Ask>> exprs;
public:
  AskCmp(std::vector<std::unique_ptr<Ask>> exprs)
    : exprs(std::move(exprs)) {}
  ~AskCmp(){}
  std::string generate() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->generate() + " && ";
    }
    res.pop_back();
    res.pop_back();
    res.pop_back();
    res += ")";
    return std::move(res);
  }
};

// Tell language: x <- [expr..expr]
class Tell {
  std::string var;
  std::unique_ptr<Expr> lb;
  std::unique_ptr<Expr> ub;
public:
  Tell(std::string var, std::unique_ptr<Expr> lb, std::unique_ptr<Expr> ub):
    var(var), lb(lb), ub(ub)
  {}
  std::string generate() const {
    return "vstore.update({(" + lb->generate() + "),(" + ub->generate() + ")})";
  }
  ~Tell();
};

// Ask => Tell
class GuardedCommand {
  std::unique_ptr<Ask> ask;
  std::unique_ptr<Tell> tell;
public:
  GuardedCommand(std::unique_ptr<Ask> ask, std::unique_ptr<Tell> tell):
    ask(ask), tell(tell) {}

  std::string generate() const {
    return "if(" + ask->generate() + ") {" + tell->generate() + "}";
  }
};

#endif // TURBO_LANG_HPP

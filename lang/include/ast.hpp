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

#include <memory>
#include <vector>
#include <string>

class Expr {
public:
  // Generate the C++ code associated with this expression.
  virtual std::string compile() const = 0;
  // Generate a representation of this expression.
  virtual std::string pretty() const = 0;
  virtual ~Expr(){}
};

static int inf = std::numeric_limits<int>::max();
static int neg_inf = std::numeric_limits<int>::min();

class IntConstant: public Expr {
  int v;
public:
  IntConstant(int v): v(v) {}
  ~IntConstant(){}
  std::string compile() const {
    return std::to_string(v);
  }

  std::string pretty() const {
    if(v == inf) {
      return "\u221E";
    }
    else if(v == neg_inf) {
      return "-\u221E";
    }
    else {
      return compile();
    }
  }
};

class BoolConstant: public Expr {
  bool v;
public:
  BoolConstant(bool v): v(v) {}
  BoolConstant(BoolConstant&&)=default;
  ~BoolConstant(){}
  std::string compile() const {
    if(v) { return "true"; }
    else { return "false"; }
  }
  std::string pretty() const {
    return compile();
  }
};

class LB : public Expr {
  std::string var;
public:
  LB(std::string var): var(var) {}
  ~LB(){}
  std::string compile() const {
    return "vstore.lb(" + var + ")";
  }
  std::string pretty() const {
    return var;
  }
};

class UB : public Expr {
  std::string var;
public:
  UB(std::string var): var(var) {}
  ~UB(){}
  std::string compile() const {
    return "vstore.ub(" + var + ")";
  }
  std::string pretty() const {
    return var;
  }
};

class BinOp: public Expr {
  std::vector<std::unique_ptr<Expr>> exprs;
  char op;
public:
  BinOp(std::vector<std::unique_ptr<Expr>> exprs, char op):
    exprs(std::move(exprs)), op(op) {}
  ~BinOp(){}
  std::string compile() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->compile() + " " + op + " ";
    }
    res.pop_back();
    res.pop_back();
    res += ")";
    return std::move(res);
  }
  std::string pretty() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->pretty() + " " + op + " ";
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
  CmpOp(CmpOp&&) = default;
  ~CmpOp(){}
  std::string compile() const {
    return left->compile() + " " + op + " " + right->compile();
  }
  std::string pretty() const {
    return left->pretty() + " " + op + " " + right->pretty();
  }
};

// ASK language: expr <op> expr / A1 /\ A2 / true / false
class Ask {
public:
  virtual std::string compile() const = 0;
  virtual std::string pretty() const = 0;
  virtual ~Ask(){}
};

class AskBool: public Ask {
  BoolConstant v;
public:
  AskBool(BoolConstant v): v(std::move(v)) {}
  AskBool(bool v): v(BoolConstant(v)) {}
  ~AskBool(){}
  std::string compile() const {
    return v.compile();
  }
  std::string pretty() const {
    return v.pretty();
  }
};

class AskCmp: public Ask {
  CmpOp expr;
public:
  AskCmp(CmpOp expr): expr(std::move(expr)) {}
  ~AskCmp(){}
  std::string compile() const {
    return expr.compile();
  }
  std::string pretty() const {
    return expr.pretty();
  }
};

class AskAnd: public Ask {
  std::vector<std::unique_ptr<Ask>> exprs;
public:
  AskAnd(std::vector<std::unique_ptr<Ask>> exprs)
    : exprs(std::move(exprs)) {}
  ~AskAnd(){}
  std::string compile() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->compile() + " && ";
    }
    res.pop_back();
    res.pop_back();
    res.pop_back();
    res += ")";
    return std::move(res);
  }
  std::string pretty() const {
    std::string res("(");
    for(auto& e: exprs) {
      res += e->pretty() + " \u2227 ";
    }
    res.pop_back();
    res.pop_back();
    res.pop_back();
    res += ")";
    return std::move(res);
  }
};

// Tell language: x <- [expr..expr] / true / false
class Tell {
public:
  virtual std::string compile() const = 0;
  virtual std::string pretty() const = 0;
  virtual ~Tell(){}
};

class TellInterval: public Tell {
  std::string var;
  std::unique_ptr<Expr> lb;
  std::unique_ptr<Expr> ub;
public:
  TellInterval(std::string var, std::unique_ptr<Expr> lb, std::unique_ptr<Expr> ub):
    var(var), lb(std::move(lb)), ub(std::move(ub))
  {}
  std::string compile() const {
    return "vstore.update(" + var + ", {" + lb->compile() + ", " + ub->compile() + "})";
  }
  std::string pretty() const {
    return /* var + " \u2190 " +*/ var + " \u2294 [" + lb->pretty() + ".." + ub->pretty() + "]";
  }
  ~TellInterval(){}
};

class TellBool: public Tell {
  BoolConstant v;
public:
  TellBool(BoolConstant v): v(std::move(v)) {}
  TellBool(bool v): v(BoolConstant(v)) {}
  ~TellBool(){}
  std::string compile() const {
    return v.compile();
  }
  std::string pretty() const {
    return v.pretty();
  }
};

static std::unique_ptr<TellInterval> make_int_interval(std::string var, int lb, int ub) {
  return std::make_unique<TellInterval>(var,
    std::make_unique<IntConstant>(lb),
    std::make_unique<IntConstant>(ub));
}

// Ask => Tell
class GuardedCommand {
  std::unique_ptr<Ask> ask;
  std::unique_ptr<Tell> tell;
public:
  GuardedCommand(std::unique_ptr<Ask> ask, std::unique_ptr<Tell> tell):
    ask(std::move(ask)), tell(std::move(tell)) {}

  std::string compile() const {
    std::string ask_code = ask->compile();
    if(ask_code != "true") {
      return "if(" + ask_code + ") {" + tell->compile() + ";}";
    }
    else {
      return tell->compile() + ";";
    }
  }

  std::string pretty() const {
    return ask->pretty() + " \u21D2 " + tell->pretty();
  }
};

static GuardedCommand make_true_gc(std::unique_ptr<Tell> tell) {
  std::unique_ptr<AskBool> ask = std::make_unique<AskBool>(true);
  return GuardedCommand(std::move(ask), std::move(tell));
}

static GuardedCommand make_false_gc() {
  std::unique_ptr<AskBool> ask = std::make_unique<AskBool>(true);
  std::unique_ptr<TellBool> tell = std::make_unique<TellBool>(false);
  return GuardedCommand(std::move(ask), std::move(tell));
}

#endif // TURBO_LANG_HPP

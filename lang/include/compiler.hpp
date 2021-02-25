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

#ifndef COMPILER_HPP
#define COMPILER_HPP

#include <vector>
#include <map>
#include <tuple>
#include <exception>

#include "XCSP3Tree.h"
#include "XCSP3TreeNode.h"
#include "XCSP3Variable.h"
#include "XCSP3Constants.h"

#include "ast.hpp"

using namespace XCSP3Core;

class Compiler {
private:
  std::vector<GuardedCommand> commands;
public:
  Compiler() = default;
  void variable(std::string name, int lb, int ub) {
    commands.push_back(make_true_gc(make_int_interval(name, lb, ub)));
  }

  void false_gc() {
    commands.push_back(make_false_gc());
  }

  void unary_constraint(std::string x, OrderType op, int k) {
    if(op == LT) {
      variable(x, neg_inf, k-1);
    }
    else if (op == LE) {
      variable(x, neg_inf, k);
    }
    else if (op == GT) {
      variable(x, k+1, inf);
    }
    else if (op == GE) {
      variable(x, k, inf);
    }
    else if(op == EQ) {
      variable(x, k, k);
    }
    else{
      throw std::runtime_error("Operators IN and NE are not supported in unary constraints.");
    }
  }

  // Transform X * 1 into X.
  void evaluate_constant(Node** node_ptr) {
    Node* node = *node_ptr;
    if(node->type == OMUL) {
      if(node->parameters[0]->type == OVAR && node->parameters[1]->type == ODECIMAL) {
        if(((NodeConstant*)node->parameters[1])->val == 1) {
          *node_ptr = node->parameters[0];
        }
      }
    }
  }

  // Treat constraint of the form X * a <= b.
  bool le_mul_domain(Node* node) {
    if (node->type == OLE &&
        node->parameters[0]->type == OMUL &&
        node->parameters[1]->type == ODECIMAL &&
        node->parameters[0]->parameters[0]->type == OVAR &&
        node->parameters[0]->parameters[1]->type == ODECIMAL)
    {
      std::string x = node->parameters[0]->parameters[0]->toString();
      int a = ((NodeConstant*)node->parameters[0]->parameters[1])->val;
      int b = ((NodeConstant*)node->parameters[1])->val;
      if(a == 0) {
        if(b < 0) {
          false_gc();
        }
      }
      else if (b == 0) {
        if(a > 0) {
          unary_constraint(x, LE, 0);
        }
        else if (a < 0) {
          unary_constraint(x, GE, 0);
        }
      }
      else {
        // At this point, a and b are different from 0.
        int res = b / a;
        if(a > 0 && b > 0) {
          unary_constraint(x, LE, res);
        }
        else if(a > 0 && b < 0) {
          unary_constraint(x, LE, res - (-b % a));
        }
        else if(a < 0 && b > 0) {
          unary_constraint(x, GE, res);
        }
        else {
          unary_constraint(x, GE, res + (-b % -a));
        }
      }
      return true;
    }
    return false;
  }

  void strengthen_domain_from_node(Node* node) {
    if (node->parameters.size() != 2) {
      throw std::runtime_error("Expected binary constraints.");
    }
    bool treated = le_mul_domain(node);
    if(!treated) {
      if (node->parameters[0]->type != OVAR) {
        evaluate_constant(&node->parameters[0]);
        if (node->parameters[0]->type != OVAR) {
          std::cout << node->toString() << std::endl;
          throw std::runtime_error("Expected variable on the lhs (in domain constraint).");
        }
      }
      if (node->parameters[1]->type != ODECIMAL) {
        throw std::runtime_error("Expected value on the rhs.");
      }
      std::string x = node->parameters[0]->toString();
      int v = dynamic_cast<NodeConstant*>(node->parameters[1])->val;
      OrderType op;
      if (node->type == OLE) { op = LE; }
      else if (node->type == OLT) { op = LT; }
      else if (node->type == OGE) { op = GE; }
      else if (node->type == OGT) { op = GT; }
      else if (node->type == OEQ) { op = EQ; }
      else if (node->type == ONE) { op = NE; }
      else if (node->type == OIN) { op = IN; }
      else {
        throw std::runtime_error("Unsupported unary domain operator.");
      }
      unary_constraint(x, op, v);
    }
  }

  // The node must have a very precise shape, X <= Y or X <= Y + k, otherwise a runtime_error is thrown.
  Propagator* make_temporal_constraint_from_node(Node* node) {
    if (node->type == OLE) {
      if (node->parameters.size() != 2) {
        throw std::runtime_error("Expected binary constraints.");
      }
      if (node->parameters[0]->type != OVAR) {
        throw std::runtime_error("Expected variable on the lhs (in temporal constraint).");
      }
      std::string x = node->parameters[0]->toString();
      if (node->parameters[1]->type == OVAR) {
        std::string y = node->parameters[1]->toString();
        return make_temporal_constraint(x, 0, LE, y);
      }
      else if (node->parameters[1]->type == OADD) {
        Node* add = node->parameters[1];
        if (add->parameters[0]->type != OVAR || add->parameters[1]->type != ODECIMAL) {
          throw std::runtime_error("Expected <var> + <constant>.");
        }
        std::string y = add->parameters[0]->toString();
        int k = dynamic_cast<NodeConstant*>(add->parameters[1])->val;
        return make_temporal_constraint(x, -k, LE, y);
      }
      else {
        std::cout << node->toString() << std::endl;
        throw std::runtime_error("Expected rhs of type OADD or OVAR.");
      }
    }
    else {
      throw std::runtime_error("Expect node in canonized form. TemporalProp constraint of the form x <= y + k");
    }
  }

  // void add_reified_constraint(Node* node) {
  //   if (node->parameters[0]->type == OVAR &&
  //       node->parameters[1]->type == OAND) {
  //     std::string b = node->parameters[0]->toString();
  //     NodeAnd* and_node = dynamic_cast<NodeAnd*>(node->parameters[1]);
  //     Propagator* p1 = make_temporal_constraint_from_node(and_node->parameters[0]);
  //     Propagator* p2 = make_temporal_constraint_from_node(and_node->parameters[1]);
  //     Propagator* rhs = new LogicalAnd(p1, p2);
  //     constraints.propagators.push_back(new ReifiedProp(std::get<0>(var2idx[b]), rhs));
  //   }
  //   else if (node->parameters[0]->type == OAND &&
  //     node->parameters[1]->type == OVAR) {
  //     std::swap(node->parameters[0], node->parameters[1]);
  //     add_reified_constraint(node);
  //   }
  //   else {
  //     throw std::runtime_error("Expected reified constraint of the form  b <=> (c1 /\\ c2)");
  //   }
  // }

  void constraint(Tree *tree) {
    if (tree->arity() == 1) {
      strengthen_domain_from_node(tree->root);
    }
    // b <=> (x < y /\ y - x >= 1)
    // else if(tree->root->type == OIFF) {
    //   add_reified_constraint(tree->root);
    // }
    // else if(tree->root->type == OLE) {
    //   add_linear_constraint(tree->root);
    // }
    else {
      // throw std::runtime_error("Unsupported constraint.");
    }
  }

  std::string compile() {
    std::string result;
    for(auto& gc: commands) {
      result = result + gc.compile() + "\n";
    }
    return result;
  }

  std::string pretty() {
    std::string result;
    for(auto& gc: commands) {
      result += gc.pretty() + "\n";
    }
    return result;
  }
};

#endif

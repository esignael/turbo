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
#include <cstdlib>

#include "solver.cuh"
#include "propagators.cuh"

#include "XCSP3CoreParser.h"

#include "XCSP3_turbo_callbacks.hpp"

void usage_and_exit(char** argv) {
    std::cout << "usage: " << argv[0] << " [timeout (seconds)] xcsp3instance.xml" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  int timeout = INT_MAX;
  int file_arg = 1;
  if(argc == 3) {
    timeout = std::atoi(argv[1]);
    if(timeout <= 0) {
      usage_and_exit(argv);
    }
    file_arg = 2;
  }
  else if (argc != 2) {
    usage_and_exit(argv);
  }

  try
  {
    ModelBuilder* model_builder = new ModelBuilder();
    XCSP3_turbo_callbacks cb(model_builder);
    XCSP3CoreParser parser(&cb);
    parser.parse(argv[file_arg]); // fileName is a string
    Constraints constraints = model_builder->build_constraints();
    VStore* vstore = model_builder->build_store();
    Var minimize_x = model_builder->build_minimize_obj();
    solve(vstore, constraints, minimize_x, timeout);
    vstore->free_names();
    vstore->~VStore();
    free2(vstore);
  }
  catch (exception &e)
  {
    cout.flush();
    cerr << "\n\tUnexpected exception:\n";
    cerr << "\t" << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}

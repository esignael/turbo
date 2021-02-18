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

#include "compiler.hpp"
#include "XCSP3CoreParser.h"

#include "XCSP3_turbo_lang_callbacks.hpp"
#include "compiler.hpp"

void usage_and_exit(char** argv) {
    std::cout << "usage: " << argv[0] << " xcsp3instance.xml" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    usage_and_exit(argv);
  }
  try
  {
    Compiler compiler;
    XCSP3_turbo_lang_callbacks cb(compiler);
    XCSP3CoreParser parser(&cb);
    parser.parse(argv[1]); // fileName is a string
    std::cout << compiler.pretty() << std::endl;
    std::cout << compiler.compile() << std::endl;
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

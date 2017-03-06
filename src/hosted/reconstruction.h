//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
#include "common.h"

#include <ebbrt/Cpu.h>
#include <ebbrt/hosted/PoolAllocator.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace ebbrt;
using namespace std;

namespace po = boost::program_options;

struct parameters PARAMETERS;

char *EXEC_NAME;

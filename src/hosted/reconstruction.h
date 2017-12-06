//    Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "../utils.h"

#include "irtkReconstruction.h"

#include <ebbrt/Cpu.h>
#include <ebbrt/hosted/PoolAllocator.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace ebbrt;
using namespace std;

namespace po = boost::program_options;

struct arguments ARGUMENTS;

size_t _FeIOCPU;

size_t _InitReconCPU;

char *EXEC_NAME;

void parseInputParameters(int argc, char **argv);

vector<irtkRigidTransformation> getTransformations(int* templateNumber);

vector<irtkRealImage> getStacks(EbbRef<irtkReconstruction> reconstruction);

void allocateBackends(EbbRef<irtkReconstruction> reconstruction);

void initializeThikness(vector<irtkRealImage> stacks);

irtkRealImage* getMask(EbbRef<irtkReconstruction> reconstruction,
    vector<irtkRealImage> &stacks, 
    vector<irtkRigidTransformation>& stackTransformations,
    int templateNumber);

void applyMask(EbbRef<irtkReconstruction> reconstruction,
    vector<irtkRealImage>& stacks, 
    vector<irtkRigidTransformation>& stackTransformations,
    int templateNumber);

void volumetricRegistration(EbbRef<irtkReconstruction> reconstruction,
    vector<irtkRealImage> stacks, 
    vector<irtkRigidTransformation>& stackTransformations,
    int templateNumber);

void eraseInputStackTuner(vector<irtkRealImage> stacks, 
    vector<irtkRigidTransformation>& stackTransformations);

void AppMain();

int main(int argc, char **argv);

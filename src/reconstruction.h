//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "irtkReconstructionEbb.h"
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace ebbrt;

namespace po = boost::program_options;

struct parameters {
  string outputName; 
  string maskName;
  string referenceVolumeName;
  vector<string> inputStacks;
  vector<string> inputTransformations;
  vector<double> thickness;
  vector<int> packages;
  int iterations; 
  double sigma;
  double resolution;
  int levels;
  double averageValue;
  double delta;
  double lambda;
  double lastIterLambda;
  double smoothMask;
  bool globalBiasCorrection;
  double lowIntensityCutoff;
  vector<int> forceExcluded;
  bool intensityMatching;
  string logId;
  bool debug;
  int recIterationsFirst;
  int recIterationsLast;
  unsigned int numInputStacksTuner;
  bool noLog;
  vector<int> devicesToUse;
  string tFolder;
  string sFolder;
  unsigned int T1PackageSize;
  unsigned int numDevicesToUse;
  bool useCPU;
  bool useCPUReg;
  bool useGPUReg;
  bool useAutoTemplate;
  bool useSINCPSF;
  bool disableBiasCorr;
  int numThreads;
  int numBackendNodes;
  int numFrontendCPUs;
};

struct parameters PARAMETERS;
char *EXEC_NAME;

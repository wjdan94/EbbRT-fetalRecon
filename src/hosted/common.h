//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef COMMON_H
#define COMMON_H
#include <string>
#include <vector>

using namespace std;

struct arguments {
  string outputName; 
  string maskName;
  string referenceVolumeName;
  string logId;
  string tFolder;
  string sFolder;

  vector<string> inputStacks;
  vector<string> inputTransformations;
  vector<double> thickness;
  vector<int> packages;
  vector<int> forceExcluded;
  vector<int> devicesToUse;

  int iterations; 
  int levels;
  int recIterationsFirst;
  int recIterationsLast;
  int numThreads;
  int numBackendNodes;
  int numFrontendCPUs;
  
  unsigned int numInputStacksTuner;
  unsigned int T1PackageSize;
  unsigned int numDevicesToUse;

  double sigma;
  double resolution;
  double averageValue;
  double delta;
  double lambda;
  double lastIterLambda;
  double smoothMask;
  double lowIntensityCutoff;

  bool globalBiasCorrection;
  bool intensityMatching;
  bool debug;
  bool noLog;
  bool useCPU;
  bool useCPUReg;
  bool useAutoTemplate;
  bool useSINCPSF;
  bool disableBiasCorr;
};

#endif // COMMON_H

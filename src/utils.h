//    Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef UTILS_H
#define UTILS_H

#define PSF_SIZE 128

#define COEFF_INIT 0
#define GAUSSIAN_RECONSTRUCTION 1
#define SIMULATE_SLICES 2
#define INITIALIZE_ROBUST_STATISTICS 3
#define E_STEP_I 4
#define E_STEP_II 5
#define E_STEP_III 6
#define SCALE 7
#define SUPERRESOLUTION 8
#define M_STEP 9
#define RESTORE_SLICE_INTENSITIES 10
#define SCALE_VOLUME 11
#define SLICE_TO_VOLUME_REGISTRATION 12
#define GATHER_TIMERS 13
#define PING 100


#define WORK_PHASES 13

#include <sys/time.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>

using namespace std;

typedef std::array<struct phase_data, WORK_PHASES> phases_data;

static const std::string PhaseNames[] = {"coeffInit",
                                         "gaussianReconstruction",
                                         "simulateSlices",
                                         "initializeRobustStatistics",
                                         "eStepI",
                                         "eStepII",
                                         "eStepIII",
                                         "scale",
                                         "superResolution",
                                         "mStep",
                                         "restoreSliceIntensities",
                                         "scaleVolume",
                                         "sliceToVolumeRegistration"};

typedef struct unsigned_three {
  unsigned int x, y, z;
} uint3;

struct POINT3D {
  short x;
  short y;
  short z;
  float value;

  template <typename Archive> void serialize(Archive& ar, 
      const unsigned int version) {
    ar & x & y & z & value;
  }  
};

struct SLICEINFO
{
  int x;        //pixel x
  int y;        //pixel y
  float value;  //value
};

typedef std::vector<POINT3D> VOXELCOEFFS;
typedef std::vector<std::vector<VOXELCOEFFS> > SLICECOEFFS;

// Struct for input arguments of reconstruction.cc
struct arguments {
  string outputName; 
  string maskName;
  string tFolder;
  string sFolder;

  vector<string> inputStacks;
  vector<string> inputTransformations;
  vector<double> thickness;
  vector<int> forceExcluded;

  int iterations; 
  int levels;
  int recIterationsFirst;
  int recIterationsLast;
  int numThreads;
  int numBackendNodes;
  int numFrontendCPUs;

  unsigned int numInputStacksTuner;
  unsigned int T1PackageSize;
  
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
  bool disableBiasCorr;
};

// Initialization parameters
struct reconstructionParameters {
  bool globalBiasCorrection;
  bool adaptive;

  int sigmaBias;
  int numThreads;
  int start;
  int end;

  int directions[13][3];

  double step;
  double sigmaSCPU;
  double sigmaS2CPU;
  double mixSCPU;
  double mixCPU;
  double lowIntensityCutoff;
};

// CoeffInit() function parameters
struct coeffInitParameters {
  int stackFactor;
  int stackIndex;

  bool debug;

  double delta;
  double lambda;
  double alpha;
  double qualityFactor;
};

// EStep() function parameters
struct eStepParameters {
  double mCPU;
  double sigmaCPU;
  double meanSCPU;
  double meanS2CPU;
  double sigmaSCPU;
  double sigmaS2CPU;
  double mixSCPU;
  double mixCPU;
  double den;
};

struct eStepReturnParameters {
  double sum; 
  double den;
  double sum2;
  double den2; 
  double maxs; 
  double mins;
  double num;
};

// MStep() function parameters
struct mStepReturnParameters {
  double sigma;
  double mix;
  double num;
  double min; 
  double max; 
};

//ScaleVOlume() function parameters
struct scaleVolumeParameters {
  double num;
  double den;
};

struct phase_data {
  float time = 0.0;
  float wait = 0.0;
  uint32_t sent = 0;
  uint32_t recv = 0;
};

struct timers {
  float coeffInit;
  float gaussianReconstruction;
  float simulateSlices;
  float initializeRobustStatistics;
  float eStepI;
  float eStepII;
  float eStepIII;
  float scale;
  float superResolution;
  float mStep;
  float restoreSliceIntensities;
  float scaleVolume;
  float sliceToVolumeRegistration;
  float totalExecutionTime;
};

inline struct timeval startTimer() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start;
}

inline float endTimer(struct timeval start) {
  struct timeval end;
  gettimeofday(&end, NULL);
  float seconds = (end.tv_sec - start.tv_sec) + 
    ((end.tv_usec - start.tv_usec) / 1000000.0);
  return seconds;
}

inline void PrintPhaseHeaders() {
  cout << ",,";
  for( auto s : PhaseNames )
    cout << s << ",";
  cout << "sum" << endl;
}

inline void PrintPhasesData(string label, phases_data pd) {
  auto tsum = 0.0;
  uint64_t dsum = 0;
  cout << label << ",time,";
  for (auto p : pd){
    cout << p.time << ",";
    tsum += p.time;
  }
  cout << tsum << endl;
  tsum = 0.0;
  cout << label << ",wait,";
  for (auto p : pd){
    cout << p.wait << ",";
    tsum += p.wait;
  }
  cout << tsum << endl;
  cout << label << ",sent,";
  for (auto p : pd){
    cout << p.sent << ",";
    dsum += p.sent;
  }
  cout << dsum << endl;
  dsum = 0;
  cout << label << ",recv,";
  for (auto p : pd){
    cout << p.recv << ",";
    dsum += p.recv;
  }
  cout << dsum << endl;
}

#endif // end of UTILS_H

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define COEFF_INIT 0
#define GAUSSIAN_RECONSTRUCTION 1
#define SIMULATE_SLICES 2
#define INITIALIZE_ROBUST_STATISTICS 3
#define E_STEP_I 4
#define E_STEP_II 5
#define E_STEP_III 6

#define ITERATION_DONE 14

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

#endif // PARAMETERS_H

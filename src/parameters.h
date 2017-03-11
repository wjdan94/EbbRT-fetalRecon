#ifndef PARAMETERS_H
#define PARAMETERS_H

#define COEFF_INIT 0
#define GAUSSIAN_RECONSTRUCTION 1
#define SIMULATE_SLICES 2
#define ITERATION_DONE 14

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

struct coeffInitParameters {
  int stackFactor;
  int stackIndex;

  bool debug;

  double delta;
  double lambda;
  double alpha;
  double qualityFactor;
};
#endif // PARAMETERS_H

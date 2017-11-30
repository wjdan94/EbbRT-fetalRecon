//    Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>

#include <ebbrt/EbbRef.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/Message.h>
#include <ebbrt/SharedEbb.h>
#include <ebbrt/SpinBarrier.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/Future.h>
#include <ebbrt/Cpu.h>

#include <ebbrt/SpinLock.h>
#include <ebbrt/native/Clock.h>

#include "../utils.h"
#include "../serialize.h"

using namespace ebbrt;
using namespace std;

class irtkReconstruction : public ebbrt::Messagable<irtkReconstruction>, public irtkObject {

  private:
    // Ebb creation parameters 
    std::unordered_map<uint32_t, ebbrt::Promise<void>> _promise_map;
    std::mutex _m;
    uint32_t _id{0};

    // Input parameters

    int _numThreads;
    int _start;
    int _end;
    int _factor;

    double _delta; 
    double _lambda; 
    double _lowIntensityCutoff; 

    bool _globalBiasCorrection; 
    bool _debug;

    // Internal parameters

    ebbrt::Promise<int> _future;

    int _sigmaBias;

    size_t _IOCPU;

    int _directions[13][3];

    double _qualityFactor;
    double _step; 
    double _sigmaSCPU;
    double _sigmaS2CPU;
    double _mixSCPU;
    double _mixCPU;
    double _maxIntensity;
    double _minIntensity;
    double _averageVolumeWeight;
    double _mCPU;
    double _sigmaCPU;

    bool _adaptive;

    vector<size_t> _workers;

    vector<float> _stackFactor;

    vector<double> _scaleCPU;
    vector<double> _sliceWeightCPU;
    vector<double> _slicePotential;

    vector<int> _stackIndex;
    vector<int> _sliceInsideCPU;
    vector<int> _voxelNum;
    vector<int> _smallSlices;

    irtkRealImage _reconstructed;
    irtkRealImage _mask;
    irtkRealImage _volumeWeights;

    vector<irtkRigidTransformation> _transformations;

    vector<irtkRealImage> _slices;
    vector<irtkRealImage> _weights;
    vector<irtkRealImage> _bias;
    vector<irtkRealImage> _simulatedSlices;
    vector<irtkRealImage> _simulatedInside;
    vector<irtkRealImage> _simulatedWeights;

    vector<SLICECOEFFS> _volcoeffs;

    // SuperResolution variables
    irtkRealImage _addon;
    irtkRealImage _confidenceMap;

    // Timer
    phases_data _phase_performance;

  public:
    // Constructor
    irtkReconstruction(ebbrt::EbbId ebbid);

    // Ebb creation functions
    static ebbrt::EbbRef<irtkReconstruction>
      Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

    static irtkReconstruction& HandleFault(ebbrt::EbbId id);

    //ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);
    void Ping(ebbrt::Messenger::NetworkId nid);

    void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
        std::unique_ptr<ebbrt::IOBuf>&& buffer);

    // Reconstruction functions
    void StoreParameters(struct reconstructionParameters parameters);

    void DefineWorkers();

    // CoeffInit functions
    void ExecuteCoeffInit(ebbrt::IOBuf::DataPointer& dp, size_t cpu);

    void CoeffInit(ebbrt::IOBuf::DataPointer& dp, size_t cpu);

    void ParallelCoeffInit();
    
    void CoeffInitBootstrap(ebbrt::IOBuf::DataPointer& dp, size_t cpu);
    
    void StoreCoeffInitParameters(ebbrt::IOBuf::DataPointer& dp);

    void InitializeEMValues();

    void InitializeEM();
    
    void ReturnFromCoeffInit(ebbrt::Messenger::NetworkId frontEndNid);

    // GaussianReconstruction function
    void ExecuteGaussianReconstruction(Messenger::NetworkId frontEndNid);

    void GaussianReconstruction();
    
    void ReturnFromGaussianReconstruction(ebbrt::Messenger::NetworkId frontEndNid);

    // SimulateSlices functions
    int ExecuteSimulateSlices(ebbrt::IOBuf::DataPointer& dp); 

    void ParallelSimulateSlices();

    int SimulateSlices(ebbrt::IOBuf::DataPointer& dp);
    
    void ReturnFromSimulateSlicest(ebbrt::Messenger::NetworkId frontEndNid);

    // RobustStatistics functions
    void ExecuteInitializeRobustStatistics(Messenger::NetworkId frontEndNid);

    void InitializeRobustStatistics(double& sigma, int& num);

    void ReturnFromInitializeRobustStatistics(double& sigma, 
        int& num, Messenger::NetworkId nid);

    // EStep function
    void StoreEStepParameters(ebbrt::IOBuf::DataPointer& dp);

    double G(double x, double s);

    double M(double m);

    void ParallelEStep(struct eStepReturnParameters& parameters);

    void ExecuteEStepI(ebbrt::IOBuf::DataPointer& dp, 
        Messenger::NetworkId frontEndNid); 

    struct eStepReturnParameters EStepI(ebbrt::IOBuf::DataPointer& dp);
    
    void ExecuteEStepII(ebbrt::IOBuf::DataPointer& dp, 
        Messenger::NetworkId frontEndNid);

    struct eStepReturnParameters EStepII(ebbrt::IOBuf::DataPointer& dp);

    void ExecuteEStepIII(ebbrt::IOBuf::DataPointer& dp, 
        Messenger::NetworkId frontEndNid);

    struct eStepReturnParameters EStepIII(ebbrt::IOBuf::DataPointer& dp);

    void ReturnFromEStepI(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);
    
    void ReturnFromEStepII(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);

    void ReturnFromEStepIII(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);

    // Scale functions
    void ExecuteScale(Messenger::NetworkId frontEndNid); 

    void ParallelScale();

    void Scale();

    // Superresolution functions
    void ExecuteSuperResolution(ebbrt::IOBuf::DataPointer& dp, 
        Messenger::NetworkId frontEndNid); 

    void ParallelSuperresolution();

    void SuperResolution(ebbrt::IOBuf::DataPointer& dp);

    void ReturnFromSuperResolution(Messenger::NetworkId nid);

    // MStep functions
    void ExecuteMStep(Messenger::NetworkId frontEndNid);

    void ParallelMStep( mStepReturnParameters& parameters);

    void MStep(mStepReturnParameters& parameters);

    void ReturnFromMStep(mStepReturnParameters& parameters,
        Messenger::NetworkId nid);

    // RestoreSliceIntensities functions
    void ExecuteRestoreSliceIntensities();

    void RestoreSliceIntensities();

    // ScaleVolume functions
    void ExecuteScaleVolume(Messenger::NetworkId frontEndNid);

    struct scaleVolumeParameters ScaleVolume();
    
    void ReturnFromScaleVolume(struct scaleVolumeParameters parameters,
        Messenger::NetworkId nid);

    // SliceToVolumeRegistration functions
    void ExecuteSliceToVolumeRegistration(ebbrt::IOBuf::DataPointer& dp, 
        Messenger::NetworkId frontEndNid); 

    void ParallelSliceToVolumeRegistration();
    
    void SliceToVolumeRegistration(ebbrt::IOBuf::DataPointer& dp);
    
    void ReturnFromSliceToVolumeRegistration(Messenger::NetworkId nid);
    
    void ReturnFrom(int fn, ebbrt::Messenger::NetworkId frontEndNid);

    void SendTimers(ebbrt::Messenger::NetworkId frontEndNid);

    // Debugging functions
    inline double SumImage(irtkRealImage img);

    inline void PrintImageSums(string s);

    inline void PrintVectorSums(vector<irtkRealImage> images, string name);
    
    inline void PrintVector(vector<double> vec, string name);

    inline void PrintVector(vector<int> vec, string name);

    inline void PrintAttributeVectorSums();
    
    void ResetOrigin(irtkGreyImage &image, irtkRigidTransformation &transformation);

    void ResetOrigin(irtkRealImage &image, irtkRigidTransformation &transformation);
};

inline double irtkReconstruction::SumImage(irtkRealImage img) {
  float sum = 0.0;
  irtkRealPixel *ap = img.GetPointerToVoxels();

  for (int j = 0; j < img.GetNumberOfVoxels(); j++) {
    sum += (float)*ap;
    ap++;
  }
  return (double)sum;
}

inline void irtkReconstruction::PrintImageSums(string s) {
  cout << fixed << s <<  " _reconstructed: " 
    << SumImage(_reconstructed) << endl;

  cout << fixed << s << " _mask: "
    << SumImage(_mask) << endl;
}

inline void irtkReconstruction::PrintVectorSums(vector<irtkRealImage> images, 
    string name) {
  for (int i = _start; i < _end; i++) {
    cout << fixed << name << "[" << i << "]: " << SumImage(images[i]) << endl;
  }
}

inline void irtkReconstruction::PrintVector(vector<double> vec, 
    string name) {
  for (int i = _start; i < _end; i++) {
    cout << fixed << name << "[" << i << "]: " << vec[i] << endl;
  }
}

inline void irtkReconstruction::PrintVector(vector<int> vec, 
    string name) {
  for (int i = _start; i < _end; i++) {
    cout << fixed << name << "[" << i << "]: " << vec[i] << endl;
  }
}

inline void irtkReconstruction::PrintAttributeVectorSums() {
  PrintVectorSums(_slices, "slices");
  //PrintVectorSums(_simulatedSlices, "simulatedSlices");
  //PrintVectorSums(_simulatedInside, "simulatedInside");
  //PrintVectorSums(_simulatedWeights, "simulatedWeights");
  PrintVectorSums(_weights, "weights");
  PrintVectorSums(_bias, "bias");
}

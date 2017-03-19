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

#include "../parameters.h"
#include "../utils.h"

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

    double _delta; 
    double _lambda; 
    double _lowIntensityCutoff; 

    bool _globalBiasCorrection; 
    bool _debug;

    // Internal parameters

    ebbrt::Promise<int> _future;

    int _sigmaBias;

    int _directions[13][3];

    double _qualityFactor;
    double _step; 
    double _sigmaSCPU;
    double _sigmaS2CPU;
    double _mixSCPU;
    double _mixCPU;
    //double _alpha;
    double _maxIntensity;
    double _minIntensity;
    double _averageVolumeWeight;
    double _mCPU;
    double _sigmaCPU;

    bool _adaptive;

    vector<float> _stackFactor;

    vector<double> _scaleCPU;
    vector<double> _sliceWeightCPU;
    vector<double> _slicePotential;

    vector<int> _stackIndex;
    vector<int> _sliceInsideCPU;
    vector<int> _voxelNum;
    //TODO: delete this since is not being used
    vector<int> _smallSlices;

    irtkRealImage _reconstructed;
    irtkRealImage _mask;
    //TODO: delete this since is not being used
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

  public:
    // Constructor
    irtkReconstruction(ebbrt::EbbId ebbid);

    // Ebb creation functions
    static ebbrt::EbbRef<irtkReconstruction>
      Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

    static irtkReconstruction& HandleFault(ebbrt::EbbId id);
    ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);

    void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
        std::unique_ptr<ebbrt::IOBuf>&& buffer);

    // Reconstruction functions
    // CoeffInit functions
    void CoeffInit(ebbrt::IOBuf::DataPointer& dp);

    void ParallelCoeffInit();
    
    void StoreParameters(struct reconstructionParameters parameters);

    void CoeffInitBootstrap(ebbrt::IOBuf::DataPointer& dp);
    
    void StoreCoeffInitParameters(ebbrt::IOBuf::DataPointer& dp);

    void InitializeEMValues();

    void InitializeEM();
    
    void ReturnFromCoeffInit(ebbrt::Messenger::NetworkId frontEndNid);

    // GaussianReconstruction function
    void GaussianReconstruction();
    
    void ReturnFromGaussianReconstruction(ebbrt::Messenger::NetworkId frontEndNid);

    // SimulateSlices functions
    void ParallelSimulateSlices();

    void SimulateSlices(ebbrt::IOBuf::DataPointer& dp);
    
    void ReturnFromSimulateSlicest(ebbrt::Messenger::NetworkId frontEndNid);

    // RobustStatistics functions
    void InitializeRobustStatistics(double& sigma, int& num);

    void ReturnFromInitializeRobustStatistics(double& sigma, 
        int& num, Messenger::NetworkId nid);

    // EStep function
    void StoreEStepParameters(ebbrt::IOBuf::DataPointer& dp);

    double G(double x, double s);

    double M(double m);

    void ParallelEStep(struct eStepReturnParameters& parameters);

    struct eStepReturnParameters EStepI(ebbrt::IOBuf::DataPointer& dp);
    
    struct eStepReturnParameters EStepII(ebbrt::IOBuf::DataPointer& dp);

    struct eStepReturnParameters EStepIII(ebbrt::IOBuf::DataPointer& dp);

    void ReturnFromEStepI(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);
    
    void ReturnFromEStepII(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);

    void ReturnFromEStepIII(struct eStepReturnParameters parameters, 
        Messenger::NetworkId nid);

    // Scale functions
    
    void ParallelScale();

    void Scale();

    void ReturnFromScale(Messenger::NetworkId nid);

    // Superresolution functions
    
    void ParallelSuperresolution();

    void SuperResolution(ebbrt::IOBuf::DataPointer& dp);

    void ReturnFromSuperResolution(Messenger::NetworkId nid);


    // MStep functions
    void ParallelMStep( mStepReturnParameters& parameters);

    void MStep(mStepReturnParameters& parameters, ebbrt::IOBuf::DataPointer& dp);

    void ReturnFromMStep(mStepReturnParameters& parameters,
        Messenger::NetworkId nid);

    // RestoreSliceIntensities functions

    void RestoreSliceIntensities();

    void ReturnFromRestoreSliceIntensities(Messenger::NetworkId nid);

    // ScaleVolume functions

    struct scaleVolumeParameters ScaleVolume();
    
    void ReturnFromScaleVolume(struct scaleVolumeParameters parameters,
        Messenger::NetworkId nid);

    //

    void ReturnFrom(int fn, ebbrt::Messenger::NetworkId frontEndNid);

    // Debugging functions
    inline double SumImage(irtkRealImage img);

    inline void PrintImageSums();

    inline void PrintVectorSums(vector<irtkRealImage> images, string name);
    
    inline void PrintVector(vector<double> vec, string name);

    inline void PrintVector(vector<int> vec, string name);

    inline void PrintAttributeVectorSums();
    
    // Serialize
    void DeserializeSlice(ebbrt::IOBuf::DataPointer& dp, irtkRealImage& tmp);

    void DeserializeTransformations(ebbrt::IOBuf::DataPointer& dp, irtkRigidTransformation& tmp);
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

inline void irtkReconstruction::PrintImageSums() {
  /*
     cout << "_externalRegistrationTargetImage: " 
     << SumImage(_externalRegistrationTargetImage) << endl; 
     */

  cout << fixed << "_reconstructed: " 
    << SumImage(_reconstructed) << endl;

  cout << fixed << "_mask: "
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

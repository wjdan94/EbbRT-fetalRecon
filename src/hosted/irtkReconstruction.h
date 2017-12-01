//    Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "../utils.h"
#include "../serialize.h"

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>

#include <ebbrt/EbbAllocator.h>
#include <ebbrt/Future.h>
#include <ebbrt/Message.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/Cpu.h>

using namespace ebbrt;

class irtkReconstruction : public ebbrt::Messagable<irtkReconstruction>, 
  public irtkObject {

  private:
    std::unordered_map<string, size_t> _frontEnd_cpus_map;   // maps str(ip) to cpu index

    // Ebb creation parameters 
    std::unordered_map<uint32_t, ebbrt::Promise<void>> _promise_map;
    std::mutex _m;
    uint32_t _id{0};

    // EbbRT-related parameters
    std::vector<ebbrt::Messenger::NetworkId> _nids;
    ebbrt::Promise<void> _backendsAllocated;

    // Input parameters
    string _outputName;  
    string _maskName; 

    vector<string> _inputStacks; 
    vector<string> _inputTransformations; 

    vector<double> _thickness; 

    vector<int> _forceExcluded; 

    int _iterations;  
    int _levels; 
    int _recIterationsFirst; 
    int _recIterationsLast; 
    int _numThreads; 
    int _numBackendNodes; 
    int _numFrontendCPUs; 

    double _sigma; 
    double _resolution; 
    double _averageValue; 
    double _delta; 
    double _lambda; 
    double _smoothingLambda; 
    double _lastIterLambda; 
    double _smoothMask; 
    double _lowIntensityCutoff; 

    bool _globalBiasCorrection; 
    bool _intensityMatching; 
    bool _debug; 
    bool _disableBiasCorr; 

    phases_data _phase_performance;
    std::vector<phases_data> _backend_performance;

    // Internal parameters
    ebbrt::Promise<void> _reconstructionDone;

    const double _sigmaFactor = 6.28;
    
    irtkRealImage _externalRegistrationTargetImage;
    irtkRealImage _reconstructed;
    irtkRealImage _mask;
    irtkRealImage _volumeWeights;

    vector<double> _scaleCPU;
    vector<double> _sliceWeightCPU;
    vector<double> _slicePotential;

    vector<float> _stackFactor;

    vector<int> _stackIndex;
    vector<int> _sliceInsideCPU;
    vector<int> _smallSlices;
    vector<int> _voxelNum;

    vector<irtkRigidTransformation> _transformations;

    vector<irtkRealImage> _slices;
    vector<irtkRealImage> _simulatedSlices;
    vector<irtkRealImage> _simulatedInside;
    vector<irtkRealImage> _simulatedWeights;
    vector<irtkRealImage> _weights;
    vector<irtkRealImage> _bias;

    vector<SLICECOEFFS> _volcoeffs;

    int _directions[13][3];

    int _sigmaBias;
    int _reconRecv;
    int _tsigma;
    int _tmix;
    int _tnum;
    uint64_t _totalBytes;
    int _received;
    int _numSum;

    double _qualityFactor;
    double _step; 
    double _sigmaSCPU;
    double _sigmaS2CPU;
    double _mixSCPU;
    double _mixCPU;
    double _alpha;
    double _tmin;
    double _tmax;
    double _maxIntensity;
    double _minIntensity;
    double _sigmaCPU;
    double _sigmaSum;
    double _mCPU;

    bool _templateCreated;
    bool _haveMask;
    bool _adaptive;

    ebbrt::Promise<int> _future;
    
    int* _imageIntPtr = NULL;
    double* _imageDoublePtr = NULL;
    // TODO: this case should be handled with the _imageDoublePtr
    // remember to fix it and delete it
    double* _volumeWeightsDoublePtr = NULL;

    // EStep() variables
    double _sum;
    double _num;
    double _den;
    double _den2;
    double _sum2;
    double _maxs;
    double _mins;
    double _meanSCPU;
    double _meanS2CPU;

    // SuperResolution() variables
    irtkRealImage _confidenceMap;
    irtkRealImage _addon;

    // MStep() variables
    // TODO: figure out if all these variables are needed
    double _mSigma;
    double _mMix;
    double _mNum;
    double _mMin;
    double _mMax;

  public:

    // Constructor
    irtkReconstruction(ebbrt::EbbId ebbid);
    
    // Ebb creation functions
    static ebbrt::EbbRef<irtkReconstruction>
      Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

    static irtkReconstruction& HandleFault(ebbrt::EbbId id);

    //ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);
    void Ping(ebbrt::Messenger::NetworkId nid);

    // EbbRT-related functions
    void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
        std::unique_ptr<ebbrt::IOBuf>&& buffer);

    // Node allocation functions
    void AddNid(ebbrt::Messenger::NetworkId nid);

    ebbrt::Future<void> WaitPool();

    ebbrt::Future<void> ReconstructionDone();

    // Reconstruction functions
    void SetParameters(arguments args);

    void SetDefaultParameters();

    void SetSmoothingParameters(double lambda);

    irtkRealImage CreateMask(irtkRealImage image);

    void MaskVolume();

    void TransformMask(irtkRealImage& image,
        irtkRealImage& mask,
        irtkRigidTransformation& transformation);

    void CropImage(irtkRealImage& image,
        irtkRealImage& mask);

    void InvertStackTransformations(
        vector<irtkRigidTransformation>& stack_transformations);

    double CreateTemplate(irtkRealImage stack,
        double resolution = 0);

    irtkRealImage GetMask();

    void SetMask(irtkRealImage * mask, double sigma, double threshold = 0.5);

    void StackRegistrations(vector<irtkRealImage>& stacks,
        vector<irtkRigidTransformation>& stack_transformations,
        int templateNumber, bool useExternalTarget = false);

    irtkRealImage CreateAverage(vector<irtkRealImage>& stacks,
        vector<irtkRigidTransformation>& stack_transformations);

    void MatchStackIntensitiesWithMasking(vector<irtkRealImage>& stacks,
        vector<irtkRigidTransformation>& stack_transformations,
        double averageValue,
        bool together = false);

    void CreateSlicesAndTransformations(vector<irtkRealImage>& stacks,
        vector<irtkRigidTransformation>& stack_transformations,
        vector<double>& thickness,
        const vector<irtkRealImage> &probability_maps = vector<irtkRealImage>()
        );

    uint64_t GetTotalBytes();

    void MaskSlices();

    void ReadTransformation(char* folder);

    void InitializeEM();

    void InitializeEMValues();

    struct reconstructionParameters CreateReconstructionParameters(
        int start, int end);

    float Gather(string fn);

    void ReturnFrom();

    // CoeffInit() function
    struct coeffInitParameters createCoeffInitParameters();

    void CoeffInitBootstrap(struct coeffInitParameters parameters);

    void CoeffInit(struct coeffInitParameters parameters);

    void CoeffInit(int iteration);

    void ReturnFromCoeffInit(ebbrt::IOBuf::DataPointer & dp);

    //GaussianReconstruction() function
    void GaussianReconstruction();

    void ExcludeSlicesWithOverlap();

    void AssembleImage(ebbrt::IOBuf::DataPointer & dp);

    void ReturnFromGaussianReconstruction(ebbrt::IOBuf::DataPointer & dp);

    //SimulateSlices() function
    void SimulateSlices(bool initialize);

    void ReturnFromSimulateSlices(ebbrt::IOBuf::DataPointer & dp);

    //InitializeRobustStatistics() function
    void InitializeRobustStatistics();

    void ReturnFromInitializeRobustStatistics(ebbrt::IOBuf::DataPointer & dp);

    //EStep() function 
    void EStepI();

    void EStepII();

    void EStepIII();

    void EStep();

    void ReturnFromEStepI(ebbrt::IOBuf::DataPointer & dp);

    void ReturnFromEStepII(ebbrt::IOBuf::DataPointer & dp);

    void ReturnFromEStepIII(ebbrt::IOBuf::DataPointer & dp);

    //Scale() function
    void Scale();

    void ReturnFromScale(ebbrt::IOBuf::DataPointer & dp);

    //SuperResolution() function
    void AdaptiveRegularization2(vector<irtkRealImage> &_b,
        vector<double> &_factor, irtkRealImage &_original);

    void AdaptiveRegularization1(vector<irtkRealImage> &_b,
        vector<double> &_factor, irtkRealImage &_original);

    void AdaptiveRegularization(int iteration, irtkRealImage &original);

    void BiasCorrectVolume(irtkRealImage &original);

    void SuperResolution(int iteration);

    void ReturnFromSuperResolution(ebbrt::IOBuf::DataPointer & dp);

    //MStep() function
    void MStep(int iteration);

    void ReturnFromMStep(ebbrt::IOBuf::DataPointer & dp);

    //RestoreSliceIntensities() function
    void RestoreSliceIntensities();

    void ReturnFromRestoreSliceIntensities( ebbrt::IOBuf::DataPointer & dp);

    //ScaleVolume() function
    void ScaleVolume();

    void ReturnFromScaleVolume(ebbrt::IOBuf::DataPointer & dp);

    //SliceToVolumeRegistration() function
    void ParallelSliceToVolumeRegistration();

    void SliceToVolumeRegistration();

    void ReturnFromSliceToVolumeRegistration(ebbrt::IOBuf::DataPointer & dp);

    // Start program execution
    void ReturnFromGatherTimers(ebbrt::IOBuf::DataPointer & dp);

    void GatherBackendTimers();

    void GatherFrontendTimers();

    void Execute();

    // Static Reconstruction functions
    static void ResetOrigin(irtkGreyImage &image, 
        irtkRigidTransformation& transformation);

    // For debugging purposes
    inline double SumImage(irtkRealImage img);

    inline void PrintImageSums(string s);

    inline void PrintVectorSums(vector<irtkRealImage> images, string name);

    inline void PrintAttributeVectorSums();
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
  cout << fixed << s << " _reconstructed: " 
       << SumImage(_reconstructed) << endl;

  cout << fixed << s << " _mask: "
       << SumImage(_mask) << endl;
}

inline void irtkReconstruction::PrintVectorSums(vector<irtkRealImage> images, 
    string name) {
  for (int i = 0; i < (int) images.size(); i++) {
    cout << fixed << name << "[" << i << "]: " << SumImage(images[i]) << endl;
  }
}

inline void irtkReconstruction::PrintAttributeVectorSums() {
  PrintVectorSums(_slices, "_slices");
  PrintVectorSums(_simulatedSlices, "_simulatedSlices");
  PrintVectorSums(_simulatedInside, "_simulatedInside");
  PrintVectorSums(_simulatedWeights, "_simulatedWeights");
  PrintVectorSums(_weights, "_weights");
  PrintVectorSums(_bias, "_bias");
}

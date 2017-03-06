
#include "common.h"

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>

#include <ebbrt/EbbAllocator.h>
#include <ebbrt/Future.h>
#include <ebbrt/Message.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticIOBuf.h>

using namespace ebbrt;

class irtkReconstruction : public ebbrt::Messagable<irtkReconstruction>, public irtkObject {

  private:

    // Ebb creation parameters 
    std::unordered_map<uint32_t, ebbrt::Promise<void>> _promise_map;
    std::mutex _m;
    uint32_t _id{0};

    // EbbRT-related parameters
    std::vector<ebbrt::Messenger::NetworkId> _nids;

    // Input parameters
    string _outputName;  // Not used
    string _maskName; // Not used
    string _referenceVolumeName; // Not used
    string _logId; // Not used
    string _tFolder; // Not used
    string _sFolder; // Not used

    vector<string> _inputStacks; // Not used
    vector<string> _inputTransformations; // Not used
    vector<double> _thickness; // Not used
    vector<int> _packages; // Not used
    vector<int> _forceExcluded; // Not used
    vector<int> _devicesToUse; // Not used

    int _iterations;  // Not used
    int _levels; // Not used
    int _recIterationsFirst; // Not used
    int _recIterationsLast; // Not used
    int _numThreads; // Not used
    int _numBackendNodes; // Not used
    int _numFrontendCPUs; // Not used

    unsigned int _numInputStacksTuner; // Not used
    unsigned int _T1PackageSize; // Not used
    unsigned int _numDevicesToUse; // Not used

    double _sigma; // Not used
    double _resolution; // Not used
    double _averageValue; // Not used
    double _delta; // Not used
    double _lambda; // Not used
    double _lastIterLambda; // Not used
    double _smoothMask; // Not used
    double _lowIntensityCutoff; // Not used

    bool _globalBiasCorrection; // Not used
    bool _intensityMatching; // Not used
    bool _debug; // Not used
    bool _noLog; // Not used
    bool _useCPU; // Not used
    bool _useCPUReg; // Not used
    bool _useAutoTemplate; // Not used
    bool _useSINCPSF; // Not used
    bool _disableBiasCorr; // Not used

    // Internal parameters
    int _directions[13][3];

    int _qualityFactor;
    int _sigmaBias;
    int _reconRecv;
    int _tsigma;
    int _tmix;
    int _tnum;
    int _totalBytes;

    double _step; 
    double _sigmaSCpu;
    double _sigmaS2Cpu;
    double _mixSCpu;
    double _mixCpu;
    double _alpha;
    double _tmin;
    double _tmax;

    bool _templateCreated;
    bool _haveMask;
    bool _adaptive;

  public:

    // Constructor
    irtkReconstruction(ebbrt::EbbId ebbid);
    
    // Ebb creation functions
    static ebbrt::EbbRef<irtkReconstruction>
      Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

    static irtkReconstruction& HandleFault(ebbrt::EbbId id);

    ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);

    // EbbRT-related functions
    void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
        std::unique_ptr<ebbrt::IOBuf>&& buffer);

    // Reconstruction functions
    void setParameters(parameters p);

    void setDefaultParameters();
};

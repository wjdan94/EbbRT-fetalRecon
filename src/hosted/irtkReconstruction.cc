
#include "irtkReconstruction.h"

#include <ebbrt/EbbRef.h>
#include <ebbrt/LocalIdMap.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

EbbRef<irtkReconstruction> irtkReconstruction::Create(EbbId id) {
  return EbbRef<irtkReconstruction>(id);
}

// This Ebb is implemented with one representative per machine
irtkReconstruction &irtkReconstruction::HandleFault(EbbId id) {
  {
    // First we check if the representative is in the LocalIdMap (using a
    // read-lock)
    LocalIdMap::ConstAccessor accessor;
    auto found = local_id_map->Find(accessor, id);
    if (found) {
      auto &rep = *boost::any_cast<irtkReconstruction *>(accessor->second);
      EbbRef<irtkReconstruction>::CacheRef(id, rep);
      return rep;
    }
  }

  irtkReconstruction *rep;
  {
    // Try to insert an entry into the LocalIdMap while holding an exclusive
    // (write) lock
    LocalIdMap::Accessor accessor;
    auto created = local_id_map->Insert(accessor, id);
    if (unlikely(!created)) {
      // We raced with another writer, use the rep it created and return
      rep = boost::any_cast<irtkReconstruction *>(accessor->second);
    } else {
      // Create a new rep and insert it into the LocalIdMap
      rep = new irtkReconstruction(id);
      accessor->second = rep;
    }
  }
  // Cache the reference to the rep in the local translation table
  EbbRef<irtkReconstruction>::CacheRef(id, *rep);
  return *rep;
}

ebbrt::Future<void> irtkReconstruction::Ping(Messenger::NetworkId nid) {
  uint32_t id;
  Promise<void> promise;
  auto ret = promise.GetFuture();
  {
    std::lock_guard<std::mutex> guard(_m);
    id = _id; // Get a new id (always even)
    _id += 2;

    bool inserted;
    // insert our promise into the hash table
    std::tie(std::ignore, inserted) =
        _promise_map.emplace(id, std::move(promise));
    assert(inserted);
  }
  // Construct and send the ping message
  auto buf = MakeUniqueIOBuf(sizeof(uint32_t));
  auto dp = buf->GetMutDataPointer();
  dp.Get<uint32_t>() = id + 1; // Ping messages are odd
  SendMessage(nid, std::move(buf));
  std::printf("Ping SetMessage\n");
  return ret;
}

void irtkReconstruction::setDefaultParameters() {
  _qualityFactor = 2;

  _step = 0.0001;
  _sigmaBias = 12;
  _sigmaSCpu = 0.025;
  _sigmaS2Cpu = 0.025;
  _mixSCpu = 0.9;
  _mixCpu = 0.9;
  _alpha = (0.05 / _lambda) * _delta * _delta;

  _templateCreated = false;
  _haveMask = false;
  _adaptive = false;

  int directions[13][3] = {{1, 0, -1}, {0, 1, -1}, {1, 1, -1}, {1, -1, -1},
                           {1, 0, 0},  {0, 1, 0},  {1, 1, 0},  {1, -1, 0},
                           {1, 0, 1},  {0, 1, 1},  {1, 1, 1},  {1, -1, 1},
                           {0, 0, 1}};
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 3; j++)
      _directions[i][j] = directions[i][j];

  _nids.clear();

  _reconRecv = 0;
  _totalBytes = 0;
  _tsigma = 0;
  _tmix = 0;
  _tnum = 0;

  _tmin = voxel_limits<irtkRealPixel>::max();
  _tmax = voxel_limits<irtkRealPixel>::min();
}

irtkReconstruction::irtkReconstruction(EbbId ebbid)
    : Messagable<irtkReconstruction>(ebbid) {
      irtkReconstruction::setDefaultParameters();
}

void irtkReconstruction::setParameters(parameters p) {
  _outputName = p.outputName;  // Not used
  _maskName = p.maskName; // Not used
  _referenceVolumeName = p.referenceVolumeName; // Not used
  _logId = p.logId; // Not used
  _tFolder = p.tFolder; // Not used
  _sFolder = p.sFolder; // Not used

  _inputStacks = p.inputStacks; // Not used
  _inputTransformations = p.inputTransformations; // Not used
  _thickness = p.thickness; // Not used
  _packages = p.packages; // Not used
  _forceExcluded = p.forceExcluded; // Not used
  _devicesToUse = p.devicesToUse; // Not used

  _iterations = p.iterations;  // Not used
  _levels = p.levels; // Not used
  _recIterationsFirst = p.recIterationsFirst; // Not used
  _recIterationsLast = p.recIterationsLast; // Not used
  _numThreads = p.numThreads; // Not used
  _numBackendNodes = p.numBackendNodes; // Not used
  _numFrontendCPUs = p.numFrontendCPUs; // Not used
  
  _numInputStacksTuner = p.numInputStacksTuner; // Not used
  _T1PackageSize = p.T1PackageSize; // Not used
  _numDevicesToUse = p.numDevicesToUse; // Not used
  
  _sigma = p.sigma; // Not used
  _resolution = p.resolution; // Not used
  _averageValue = p.averageValue; // Not used
  _delta = p.delta; // Not used
  _lambda = p.lambda; // Not used
  _lastIterLambda = p.lastIterLambda; // Not used
  _smoothMask = p.smoothMask; // Not used
  _lowIntensityCutoff = p.lowIntensityCutoff; // Not used
  
  _globalBiasCorrection = p.globalBiasCorrection; // Not used
  _intensityMatching = p.intensityMatching; // Not used
  _debug = p.debug; // Not used
  _noLog = p.noLog; // Not used
  _useCPU = p.useCPU; // Not used
  _useCPUReg = p.useCPUReg; // Not used
  _useAutoTemplate = p.useAutoTemplate; // Not used
  _useSINCPSF = p.useSINCPSF; // Not used
  _disableBiasCorr = p.disableBiasCorr; // Not used
}

void irtkReconstruction::ReceiveMessage(Messenger::NetworkId nid,
    std::unique_ptr<IOBuf> &&buffer) {

}

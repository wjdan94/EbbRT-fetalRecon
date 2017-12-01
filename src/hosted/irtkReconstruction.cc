//    Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "irtkReconstruction.h"

#include <irtkResampling.h>
// Warning: The following three libraries must be imported toghether
//          in this exact order.
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>

#include <ebbrt/EbbRef.h>
#include <ebbrt/LocalIdMap.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, irtkReconstruction);

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

void irtkReconstruction::Ping(Messenger::NetworkId nid) {

  // Construct and send the ping message
  auto buf = MakeUniqueIOBuf(sizeof(int));
  auto dp = buf->GetMutDataPointer();

  dp.Get<int>() = PING;

  cout << "Sending Ping from CPU: " << ebbrt::Cpu::GetMine() << endl; 
  SendMessage(nid, std::move(buf));
}

void irtkReconstruction::SetDefaultParameters() {
  _qualityFactor = 2;

  _step = 0.0001;
  _sigmaBias = 12;
  _sigmaSCPU = 0.025;
  _sigmaS2CPU = 0.025;
  _mixSCPU = 0.9;
  _mixCPU = 0.9;

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
    irtkReconstruction::SetDefaultParameters();
  }

void irtkReconstruction::SetParameters(arguments args) {
  _outputName = args.outputName;  // Not used
  _maskName = args.maskName; // Not used

  _inputStacks = args.inputStacks; // Not used
  _inputTransformations = args.inputTransformations; // Not used
  _thickness = args.thickness; // Not used
  _forceExcluded = args.forceExcluded; // Not used

  _iterations = args.iterations;  
  _levels = args.levels; // Not used
  _recIterationsFirst = args.recIterationsFirst; 
  _recIterationsLast = args.recIterationsLast; 
  _numThreads = args.numThreads; // Not used
  _numBackendNodes = args.numBackendNodes; 
  _numFrontendCPUs = args.numFrontendCPUs; // Not used

  _sigma = (args.sigma) > 0 ? args.sigma : 20;
  _resolution = args.resolution; // Not used
  _averageValue = args.averageValue; // Not used
  _delta = args.delta; 
  _smoothingLambda = args.lambda;
  _lambda = args.lambda; 
  _alpha = (0.05 / _lambda) * _delta * _delta;
  _lastIterLambda = args.lastIterLambda; // Not used
  _smoothMask = args.smoothMask; // Not used
  _lowIntensityCutoff = (args.lowIntensityCutoff > 1) ? 1 : 0; // Not used

  _globalBiasCorrection = args.globalBiasCorrection; 
  _intensityMatching = args.intensityMatching; 
  _debug = args.debug; 
  _disableBiasCorr = args.disableBiasCorr; // Not used
}

/*
 * Pool Allocator helper functions
 */
ebbrt::Future<void> irtkReconstruction::WaitPool() {
  return std::move(_backendsAllocated.GetFuture());
}

void irtkReconstruction::AddNid(ebbrt::Messenger::NetworkId nid) {
  _nids.push_back(nid);

  cout << "Adding a network id, working on CPU " << ebbrt::Cpu::GetMine() << endl;

  int cpu_num = ebbrt::Cpu::GetPhysCpus();
  auto index = (cpu_num - (int)_nids.size()) % cpu_num;
  auto cpu_i = ebbrt::Cpu::GetByIndex(index);
  auto ctxt = cpu_i->get_context();

  cout << "CPU: " << index << " was reserved for " << nid.ToString() << endl;  

  _frontEnd_cpus_map.insert(make_pair(nid.ToString(), index));
 
  ebbrt::event_manager->SpawnRemote([this, nid]() { Ping(nid); }, ctxt);
 
  if ((int) _nids.size() == _numBackendNodes) {
    _backendsAllocated.SetValue();
  }
}


/*
 * Fetal Reconstruction functions
 */ 

ebbrt::Future<void> irtkReconstruction::ReconstructionDone() {
  return std::move(_reconstructionDone.GetFuture());
}

void irtkReconstruction::ReturnFrom() {
  _received++;
  if (_received == _numBackendNodes) {
    _received = 0;
    _future.SetValue(1);
  }
}

void irtkReconstruction::AssembleImage(ebbrt::IOBuf::DataPointer & dp) { 

  int start = dp.Get<int>();
  int end = dp.Get<int>();

  _imageIntPtr = (int *) malloc((end - start)*sizeof(int));

  dp.Get((end-start) * sizeof(int), (uint8_t*) _imageIntPtr);

  memcpy(_voxelNum.data() + start, _imageIntPtr, 
      (end-start) * sizeof(int));
  
  int size = dp.Get<int>();

  _imageDoublePtr = (double*) malloc (
      _reconstructed.GetSizeMat()*sizeof(double));

  dp.Get(size*sizeof(double), (uint8_t*) _imageDoublePtr);

  _reconstructed.SumVec(_imageDoublePtr);

  size = dp.Get<int>();

  _volumeWeightsDoublePtr = (double*) malloc (
      _volumeWeights.GetSizeMat()*sizeof(double));

  dp.Get(size*sizeof(double), (uint8_t*) _volumeWeightsDoublePtr);

  _volumeWeights.SumVec(_volumeWeightsDoublePtr);
}

void irtkReconstruction::ReturnFromGaussianReconstruction(
    ebbrt::IOBuf::DataPointer & dp) {
  AssembleImage(dp);
  ReturnFrom();
}

void irtkReconstruction::ReturnFromCoeffInit(ebbrt::IOBuf::DataPointer & dp) {
  ReturnFrom();
}

void irtkReconstruction::ReturnFromSimulateSlices(
    ebbrt::IOBuf::DataPointer & dp) {
  ReturnFrom();
}

void irtkReconstruction::ReturnFromInitializeRobustStatistics(
    ebbrt::IOBuf::DataPointer & dp) {
  int num = dp.Get<int>();
  double sigma = dp.Get<double>();
  _sigmaSum += sigma;
  _numSum += num;
  ReturnFrom();
}

void irtkReconstruction::ReturnFromEStepI(ebbrt::IOBuf::DataPointer & dp) {

  auto parameters = dp.Get<struct eStepReturnParameters>();

  _sum += parameters.sum;
  _den += parameters.den;
  _den2 += parameters.den2;
  _sum2 += parameters.sum2;
  _maxs = (_maxs > parameters.maxs) ? _maxs : parameters.maxs;
  _mins = (_mins < parameters.mins) ? _mins : parameters.mins;
  
  ReturnFrom();
}

void irtkReconstruction::ReturnFromEStepII(ebbrt::IOBuf::DataPointer & dp) {

  auto parameters = dp.Get<struct eStepReturnParameters>();

  _sum += parameters.sum;
  _den += parameters.den;
  _den2 += parameters.den2;
  _sum2 += parameters.sum2;

  ReturnFrom();
}

void irtkReconstruction::ReturnFromEStepIII(ebbrt::IOBuf::DataPointer & dp) {

  auto parameters = dp.Get<struct eStepReturnParameters>();

  _sum += parameters.sum;
  _num += parameters.num;

  ReturnFrom();
}

void irtkReconstruction::ReturnFromScale(ebbrt::IOBuf::DataPointer & dp) {
  ReturnFrom();
}

void irtkReconstruction::ReturnFromSuperResolution(
    ebbrt::IOBuf::DataPointer & dp) {
  // Read addon image
  int addonSize = dp.Get<int>();
  dp.Get(addonSize*sizeof(double), (uint8_t*) _imageDoublePtr);
  _addon.SumVec(_imageDoublePtr);

  // Read confidenceMap image
  int confidenceMapSize = dp.Get<int>();
  dp.Get(addonSize*sizeof(double), (uint8_t*) _imageDoublePtr);
  _confidenceMap.SumVec(_imageDoublePtr);

  ReturnFrom();
}

void irtkReconstruction::ReturnFromMStep(ebbrt::IOBuf::DataPointer & dp) {

  auto parameters = dp.Get<struct mStepReturnParameters>();
  _mSigma += parameters.sigma;
  _mMix += parameters.mix;
  _mNum += parameters.num;
  _mMax = (_mMax > parameters.max) ? _mMax : parameters.max;
  _mMin = (_mMin < parameters.min) ? _mMin : parameters.min;

  ReturnFrom();
}

void irtkReconstruction::ReturnFromRestoreSliceIntensities(
    ebbrt::IOBuf::DataPointer & dp) {
  ReturnFrom();
}

void irtkReconstruction::ReturnFromScaleVolume(
    ebbrt::IOBuf::DataPointer & dp) {

  auto parameters = dp.Get<struct scaleVolumeParameters>();
  _num += parameters.num;
  _den += parameters.den;
  ReturnFrom();
}

void irtkReconstruction::ReturnFromSliceToVolumeRegistration(
    ebbrt::IOBuf::DataPointer & dp) {
  int start = dp.Get<int>();
  int end = dp.Get<int>();

  for(int i = start; i < end; i++) {
    deserializeTransformations(dp, _transformations[i]);
  }

  ReturnFrom();
}

void irtkReconstruction::ReturnFromGatherTimers(
    ebbrt::IOBuf::DataPointer & dp) {
  auto phases = dp.Get<phases_data>();
  _backend_performance.emplace_back(phases);
  ReturnFrom();
}

irtkRealImage irtkReconstruction::CreateMask(irtkRealImage image) {
  // [fetalRecontruction] binarize mask
  irtkRealPixel *ptr = image.GetPointerToVoxels();
  for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
    if (*ptr > 0.0)
      *ptr = 1;
    else
      *ptr = 0;
    ptr++;
  }
  return image;
}

void irtkReconstruction::TransformMask(
    irtkRealImage &image, irtkRealImage &mask,
    irtkRigidTransformation &transformation) {

  // [fetalRecontruction] transform mask to the space of image
  irtkImageTransformation imagetransformation;
  irtkNearestNeighborInterpolateImageFunction interpolator;
  imagetransformation.SetInput(&mask, &transformation);
  irtkRealImage m = image;
  imagetransformation.SetOutput(&m);
  // [fetalRecontruction] target contains zeros and ones image, need padding -1
  imagetransformation.PutTargetPaddingValue(-1);
  // [fetalRecontruction] need to fill voxels in target where there is no 
  // [fetalRecontruction] info from source with zeroes
  imagetransformation.PutSourcePaddingValue(0);
  imagetransformation.PutInterpolator(&interpolator);
  imagetransformation.Run();
  mask = m;
}

void irtkReconstruction::CropImage(irtkRealImage &image,
    irtkRealImage &mask) {
  // [fetalReconstruction] Crops the image according to the mask

  int i, j, k;
  // [fetalReconstruction] ROI boundaries
  int x1, x2, y1, y2, z1, z2;

  // [fetalReconstruction] Original ROI
  x1 = 0;
  y1 = 0;
  z1 = 0;
  x2 = image.GetX();
  y2 = image.GetY();
  z2 = image.GetZ();

  // [fetalReconstruction] upper boundary for z coordinate
  int sum = 0;
  for (k = image.GetZ() - 1; k >= 0; k--) {
    sum = 0;
    for (j = image.GetY() - 1; j >= 0; j--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  z2 = k;

  // [fetalReconstruction] lower boundary for z coordinate
  sum = 0;
  for (k = 0; k <= image.GetZ() - 1; k++) {
    sum = 0;
    for (j = image.GetY() - 1; j >= 0; j--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  z1 = k;

  // [fetalReconstruction] upper boundary for y coordinate
  sum = 0;
  for (j = image.GetY() - 1; j >= 0; j--) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  y2 = j;

  // [fetalReconstruction] lower boundary for y coordinate
  sum = 0;
  for (j = 0; j <= image.GetY() - 1; j++) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  y1 = j;

  // [fetalReconstruction] upper boundary for x coordinate
  sum = 0;
  for (i = image.GetX() - 1; i >= 0; i--) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (j = image.GetY() - 1; j >= 0; j--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  x2 = i;

  // [fetalReconstruction] lower boundary for x coordinate
  sum = 0;
  for (i = 0; i <= image.GetX() - 1; i++) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (j = image.GetY() - 1; j >= 0; j--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }

  x1 = i;
  if (_debug)
    cout << "Region of interest is " << x1 << " " << y1 << " " << z1 << " "
      << " " << x2 << " " << y2 << " " << z2 << endl;

  // [fetalReconstruction] Cut region of interest
  image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
}

void irtkReconstruction::InvertStackTransformations(
    vector<irtkRigidTransformation> &stackTransformations) {
  for (unsigned int i = 0; i < stackTransformations.size(); i++) {
    // [fetalReconstruction] invert transformation for the stacks
    stackTransformations[i].Invert();
    stackTransformations[i].UpdateParameter();
  }
}

double irtkReconstruction::CreateTemplate(irtkRealImage stack,
    double resolution) {

  double dx, dy, dz, d;

  // [fetalRecontruction] Get image attributes - image size and voxel size
  irtkImageAttributes attr = stack.GetImageAttributes();

  // [fetalRecontruction] enlarge stack in z-direction in case top of the head 
  // is cut off
  attr._z += 2;

  // [fetalRecontruction] create enlarged image
  irtkRealImage enlarged(attr);

  // [fetalRecontruction] determine resolution of volume to reconstruct
  if (resolution <= 0) {
    // [fetalRecontruction] resolution was not given by user
    // [fetalRecontruction] set it to min of res in x or y direction
    stack.GetPixelSize(&dx, &dy, &dz);
    if ((dx <= dy) && (dx <= dz))
      d = dx;
    else if (dy <= dz)
      d = dy;
    else
      d = dz;
  } else
    d = resolution;

  // [fetalRecontruction] resample "enlarged" to resolution "d"
  irtkNearestNeighborInterpolateImageFunction interpolator;
  irtkResampling<irtkRealPixel> resampling(d, d, d);
  resampling.SetInput(&enlarged);
  resampling.SetOutput(&enlarged);
  resampling.SetInterpolator(&interpolator);
  resampling.Run();

  // [fetalRecontruction] initialize recontructed volume
  _reconstructed = enlarged;
  _templateCreated = true;

  // [fetalRecontruction] return resulting resolution of the template image
  return d;
}

irtkRealImage irtkReconstruction::GetMask() {
  return _mask;
}

void irtkReconstruction::SetMask(irtkRealImage *mask, double sigma,
    double threshold) {

  if (!_templateCreated) {
    cerr << "Please create the template before setting the mask, so that the "
      "mask can be resampled to the correct dimensions."
      << endl;
    exit(1);
  }

  _mask = _reconstructed;

  if (mask != NULL) {
    // [fetalRecontruction] if sigma is nonzero first smooth the mask
    if (sigma > 0) {
      // [fetalRecontruction] blur mask
      irtkGaussianBlurring<irtkRealPixel> gb(sigma);
      gb.SetInput(mask);
      gb.SetOutput(mask);
      gb.Run();

      // [fetalRecontruction] binarize mask
      irtkRealPixel *ptr = mask->GetPointerToVoxels();
      for (int i = 0; i < mask->GetNumberOfVoxels(); i++) {
        if (*ptr > threshold)
          *ptr = 1;
        else
          *ptr = 0;
        ptr++;
      }
    }

    // [fetalRecontruction] resample the mask according to the template volume 
    // using identity transformation
    irtkRigidTransformation transformation;
    irtkImageTransformation imagetransformation;
    irtkNearestNeighborInterpolateImageFunction interpolator;
    imagetransformation.SetInput(mask, &transformation);
    imagetransformation.SetOutput(&_mask);
    // [fetalRecontruction] target is zero image, need padding -1
    imagetransformation.PutTargetPaddingValue(-1);
    // [fetalRecontruction] need to fill voxels in target where there is no 
    // info from source with zeroes
    imagetransformation.PutSourcePaddingValue(0);
    imagetransformation.PutInterpolator(&interpolator);
    imagetransformation.Run();
  } else {
    // [fetalRecontruction] fill the mask with ones
    _mask = 1;
  }
  // [fetalRecontruction] set flag that mask was created
  _haveMask = true;
}

class ParallelStackRegistrations {
  irtkReconstruction *reconstructor;
  vector<irtkRealImage> &stacks;
  vector<irtkRigidTransformation> &stack_transformations;
  int templateNumber;
  irtkGreyImage &target;
  irtkRigidTransformation &offset;
  bool _externalTemplate;
  int nt;

  public:
  ParallelStackRegistrations(
      irtkReconstruction *_reconstructor, vector<irtkRealImage> &_stacks,
      vector<irtkRigidTransformation> &_stack_transformations,
      int _templateNumber, irtkGreyImage &_target,
      irtkRigidTransformation &_offset, int _nt, bool externalTemplate = false)
    : reconstructor(_reconstructor), stacks(_stacks),
    stack_transformations(_stack_transformations), target(_target),
    offset(_offset) {
      templateNumber = _templateNumber, _externalTemplate = externalTemplate,
      nt = _nt;
    }

  void operator()(const blocked_range<size_t> &r) const {
    for (size_t i = r.begin(); i != r.end(); ++i) {

      // [fetalRecontruction] do not perform registration for template
      if (i == templateNumber)
        continue;

      // [fetalRecontruction] rigid registration object
      irtkImageRigidRegistrationWithPadding registration;
      // [fetalRecontruction] irtkRigidTransformation 
      // [fetalRecontruction] transformation = stack_transformations[i];

      // [fetalRecontruction] set target and source (need to be converted to 
      // irtkGreyImage)
      irtkGreyImage source = stacks[i];

      // [fetalRecontruction] include offset in trasformation
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = stack_transformations[i].GetMatrix();
      m = m * mo;
      stack_transformations[i].PutMatrix(m);

      // [fetalRecontruction] perform rigid registration
      registration.SetInput(&target, &source);
      registration.SetOutput(&stack_transformations[i]);
      if (_externalTemplate) {
        registration.GuessParameterThickSlicesNMI();
      } else {
        registration.GuessParameterThickSlices();
      }
      registration.SetTargetPadding(0);
      registration.Run();

      mo.Invert();
      m = stack_transformations[i].GetMatrix();
      m = m * mo;
      stack_transformations[i].PutMatrix(m);
    }
  }

  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<size_t>(0, stacks.size()), *this);
    init.terminate();
  }
};

void irtkReconstruction::StackRegistrations(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stackTransformations, int templateNumber,
    bool useExternalTarget) {

  InvertStackTransformations(stackTransformations);

  // [fetalRecontruction] template is set as the target
  irtkGreyImage target;
  if (!useExternalTarget) {
    target = stacks[templateNumber];
  } else {
    target = _externalRegistrationTargetImage;
  }

  // [fetalRecontruction] target needs to be masked before registration
  if (_haveMask) {
    double x, y, z;
    for (int i = 0; i < target.GetX(); i++)
      for (int j = 0; j < target.GetY(); j++)
        for (int k = 0; k < target.GetZ(); k++) {
          // [fetalRecontruction] image coordinates of the target
          x = i;
          y = j;
          z = k;
          //[fetalRecontruction] change to world coordinates
          target.ImageToWorld(x, y, z);
          // [fetalRecontruction] change to mask image coordinates - 
          // [fetalRecontruction] mask is aligned with target
          _mask.WorldToImage(x, y, z);
          x = round(x);
          y = round(y);
          z = round(z);
          // [fetalRecontruction] if the voxel is outside mask ROI set it to -1 
          // [fetalRecontruction] (padding value)
          if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) &&
              (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
            if (_mask(x, y, z) == 0)
              target(i, j, k) = 0;
          } else
            target(i, j, k) = 0;
        }
  }

  irtkRigidTransformation offset;
  ResetOrigin(target, offset);

  // [fetalRecontruction] register all stacks to the target
  ParallelStackRegistrations registration(this, stacks, stackTransformations,
      templateNumber, target, offset,
      _numThreads, useExternalTarget);

  registration();
  InvertStackTransformations(stackTransformations);
}

void irtkReconstruction::MatchStackIntensitiesWithMasking(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations, double averageValue,
    bool together) {
  if (_debug)
    cout << "Matching intensities of stacks. ";

  // [fetalRecontruction] Calculate the averages of intensities for all stacks
  double sum, num;
  char buffer[256];
  unsigned int ind;
  int i, j, k;
  double x, y, z;
  vector<double> stack_average;
  irtkRealImage m;

  // [fetalRecontruction] remember the set average value
  _averageValue = averageValue;

  // [fetalRecontruction] averages need to be calculated only in ROI
  for (ind = 0; ind < stacks.size(); ind++) {
    m = stacks[ind];
    sum = 0;
    num = 0;
    for (i = 0; i < stacks[ind].GetX(); i++)
      for (j = 0; j < stacks[ind].GetY(); j++)
        for (k = 0; k < stacks[ind].GetZ(); k++) {
          // [fetalRecontruction] image coordinates of the stack voxel
          x = i;
          y = j;
          z = k;
          // [fetalRecontruction] change to world coordinates
          stacks[ind].ImageToWorld(x, y, z);
          // [fetalRecontruction] transform to template (and also _mask) space
          stack_transformations[ind].Transform(x, y, z);
          // [fetalRecontruction] change to mask image coordinates 
          // [fetalRecontruction] - mask is aligned with template
          _mask.WorldToImage(x, y, z);
          x = round(x);
          y = round(y);
          z = round(z);
          // [fetalRecontruction] if the voxel is inside mask ROI include it
          if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) &&
              (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
            if (_mask(x, y, z) == 1) {
              m(i, j, k) = 1;
              sum += stacks[ind](i, j, k);
              num++;
            } else
              m(i, j, k) = 0;
          }
        }
    // [fetalRecontruction] calculate average for the stack
    if (num > 0)
      stack_average.push_back(sum / num);
    else {
      cerr << "Stack " << ind << " has no overlap with ROI" << endl;
      exit(1);
    }
  }

  double global_average;
  if (together) {
    global_average = 0;
    for (i = 0; i < stack_average.size(); i++)
      global_average += stack_average[i];
    global_average /= stack_average.size();
  }

  if (_debug) {
    cout << "Stack average intensities are ";
    for (ind = 0; ind < stack_average.size(); ind++)
      cout << stack_average[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }

  // [fetalRecontruction] Rescale stacks
  irtkRealPixel *ptr;
  double factor;
  for (ind = 0; ind < stacks.size(); ind++) {
    if (together) {
      factor = averageValue / global_average;
      _stackFactor.push_back(factor);
    } else {
      factor = averageValue / stack_average[ind];
      _stackFactor.push_back(factor);
    }

    ptr = stacks[ind].GetPointerToVoxels();
    for (i = 0; i < stacks[ind].GetNumberOfVoxels(); i++) {
      if (*ptr > 0)
        *ptr *= factor;
      ptr++;
    }
  }

  if (_debug) {
    cout << "Slice intensity factors are ";
    for (ind = 0; ind < stack_average.size(); ind++)
      cout << _stackFactor[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }
}

void irtkReconstruction::CreateSlicesAndTransformations(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations,
    vector<double> &thickness, const vector<irtkRealImage> &probability_maps) {

  std::vector<uint3> stack_sizes_;
  uint3 temp;

  // [fetalRecontruction] for each stack
  for (unsigned int i = 0; i < stacks.size(); i++) {
    // [fetalRecontruction] image attributes contain image and voxel size
    irtkImageAttributes attr = stacks[i].GetImageAttributes();
    temp.x = attr._x;
    temp.y = attr._y;
    temp.z = attr._z;
    stack_sizes_.push_back(temp);
    // [fetalRecontruction] attr._z is number of slices in the stack
    for (int j = 0; j < attr._z; j++) {
      // [fetalRecontruction] create slice by selecting the appropreate 
      // [fetalRecontruction] region of the stack
      irtkRealImage slice =
        stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
      // [fetalRecontruction] set correct voxel size in the stack. 
      // [fetalRecontruction] Z size is equal to slice thickness.
      slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
      // [fetalRecontruction] remember the slice
      _slices.push_back(slice);
      _simulatedSlices.push_back(slice);
      _simulatedWeights.push_back(slice);
      _simulatedInside.push_back(slice);
      // [fetalRecontruction] remeber stack index for this slice
      _stackIndex.push_back(i);
      // [fetalRecontruction] initialize slice transformation with 
      // [fetalRecontruction] the stack transformation
      _transformations.push_back(stack_transformations[i]);
    }
  }
}

void irtkReconstruction::MaskSlices() {

  double x, y, z;
  int i, j;

  // [fetalRecontruction] Check whether we have a mask
  if (!_haveMask) {
    cout << "Could not mask slices because no mask has been set." << endl;
    return;
  }

  // [fetalRecontruction] mask slices
  for (int unsigned inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    irtkRealImage &slice = _slices[inputIndex];
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++) {
        // [fetalRecontruction] if the value is smaller than 1 assume it is 
        // padding
        if (slice(i, j, 0) < 0.01)
          slice(i, j, 0) = -1;
        // [fetalRecontruction] image coordinates of a slice voxel
        x = i;
        y = j;
        z = 0;
        // [fetalRecontruction] change to world coordinates in slice space
        slice.ImageToWorld(x, y, z);
        // [fetalRecontruction] world coordinates in volume space
        _transformations[inputIndex].Transform(x, y, z);
        // [fetalRecontruction] image coordinates in volume space
        _mask.WorldToImage(x, y, z);
        x = round(x);
        y = round(y);
        z = round(z);
        // [fetalRecontruction] if the voxel is outside mask ROI set it to 
        // [fetalRecontruction] -1 (padding value)
        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) &&
            (z >= 0) && (z < _mask.GetZ())) {
          if (_mask(x, y, z) == 0)
            slice(i, j, 0) = -1;
        } else
          slice(i, j, 0) = -1;
      }
  }
}

uint64_t irtkReconstruction::GetTotalBytes() {
    return _totalBytes;
}

void irtkReconstruction::ReadTransformation(char *folder) {
  int n = _slices.size();
  char name[256];
  char path[256];
  irtkTransformation *transformation;
  irtkRigidTransformation *rigidTransf;

  if (n == 0) {
    cerr << "Please create slices before reading transformations!" << endl;
    exit(1);
  }

  _transformations.clear();
  for (int i = 0; i < n; i++) {
    if (folder != NULL) {
      sprintf(name, "/transformation%i.dof", i);
      strcpy(path, folder);
      strcat(path, name);
    } else {
      sprintf(path, "transformation%i.dof", i);
    }
    transformation = irtkTransformation::New(path);
    rigidTransf = dynamic_cast<irtkRigidTransformation *>(transformation);
    _transformations.push_back(*rigidTransf);
    delete transformation;
    cout << path << endl;
  }
}

void irtkReconstruction::InitializeEM() {
  _weights.clear();
  _bias.clear();
  _scaleCPU.clear();
  _sliceWeightCPU.clear();
  _slicePotential.clear();

  for (int i = 0; i < _slices.size(); i++) {
    // [fetalRecontruction] Create images for voxel weights and bias fields
    _weights.push_back(_slices[i]);
    _bias.push_back(_slices[i]);

    // [fetalRecontruction] Create and initialize scales
    _scaleCPU.push_back(1);

    // [fetalRecontruction] Create and initialize slice weights
    _sliceWeightCPU.push_back(1);

    _slicePotential.push_back(0);
  }

  // [fetalRecontruction] Find the range of intensities
  _maxIntensity = voxel_limits<irtkRealPixel>::min();
  _minIntensity = voxel_limits<irtkRealPixel>::max();
  for (unsigned int i = 0; i < _slices.size(); i++) {
    // [fetalRecontruction] to update minimum we need to exclude padding value
    irtkRealPixel *ptr = _slices[i].GetPointerToVoxels();
    for (int ind = 0; ind < _slices[i].GetNumberOfVoxels(); ind++) {
      if (*ptr > 0) {
        if (*ptr > _maxIntensity)
          _maxIntensity = *ptr;
        if (*ptr < _minIntensity)
          _minIntensity = *ptr;
      }
      ptr++;
    }
  }
}

float irtkReconstruction::Gather(string fn) {
  auto t = startTimer();
  _future = ebbrt::Promise<int>();
  auto f = _future.GetFuture();
  if (_debug)
    cout << fn << "(): Blocking" << endl;

  f.Block();
  if (_debug)
    cout << fn << "(): Returned from future" << endl;
  return endTimer(t);
}

void irtkReconstruction::MaskVolume() {
  irtkRealPixel *pr = _reconstructed.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*pm == 0)
      *pr = -1;
    pm++;
    pr++;
  }

  if (_debug)
    PrintImageSums("[MaskVolume output]");
}

void irtkReconstruction::GatherFrontendTimers() {
  PrintPhasesData("fe", _phase_performance);
}

void irtkReconstruction::GatherBackendTimers() {

  cout << "In GatherBackendTimers()" << endl;

  for (int i = 0; i < (int) _numBackendNodes; i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = GATHER_TIMERS; 

    _totalBytes += buf->ComputeChainDataLength();

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  Gather("GatherBackendTimers");

  auto cnt = 0;
  for( auto b : _backend_performance){
    PrintPhasesData("be_"+std::to_string(cnt), b);
    cnt++;
  }
}

void irtkReconstruction::Execute() {

  cout << "In Execute() on CPU: " << ebbrt::Cpu::GetMine() << endl;
  auto start = startTimer();

  int recIterations;

  for (int it = 0; it < _iterations; it++) {
    if (_debug)
      cout << "[Iteration " << it << "] " << endl;

    if (it > 0) {
      SliceToVolumeRegistration();
    }

    auto lastIteration = it == (_iterations - 1);

    if (lastIteration) {
      SetSmoothingParameters(_lastIterLambda);
    } else {
      double lambda = _smoothingLambda;
      for (int i = 0; i < _levels; i ++) {
        if (it == _iterations * (_levels - i - 1) / _levels)
          SetSmoothingParameters(lambda);
        lambda *= 2;
      }
    }

    // Use faster reconstruction for iterations, slower for final reconstruction
    _qualityFactor = lastIteration ? 2 : 1;

    InitializeEMValues();

    CoeffInit(it);

    GaussianReconstruction();

    SimulateSlices(true);

    InitializeRobustStatistics();

    EStep();
    
    // Set number of reconstruction iterations
    if (lastIteration) 
      recIterations = _recIterationsLast;
    else
      recIterations = _recIterationsFirst;

    for (int recIt = 0; recIt < recIterations; recIt++) {

      if (_debug) {
        cout << "[Reconstruction iteration " << recIt << "]" << endl;
        cout << "[Bias input] _intensityMatching: " << _intensityMatching << endl;
        cout << "[Bias input] _disableBiasCorr: " << _disableBiasCorr << endl;
        cout << "[Bias input] _sigma: " << _sigma << endl;
      }

      if (_intensityMatching) {
        if (!_disableBiasCorr) {
          //TODO: implement Bias() function
          //if (_sigma > 0) 
            //Bias();
        }
        Scale();
      }

      SuperResolution(recIt + 1);

      if (_intensityMatching) {
        if (!_disableBiasCorr) {
          //TODO: implement NormalizeBias() function
          //if (_sigma > 0 && !_globalBiasCorrection) 
            //NormalizeBias(it);
        }
      }

      SimulateSlices(false);

      MStep(recIt + 1);

      EStep();
    
    }
    if (_debug) {
      cout << endl;
      PrintImageSums("[End of inner loop]");
    }
    
    MaskVolume();
    // TODO: Do we need to implement Evaluate()
  }

  RestoreSliceIntensities();

  ScaleVolume();

  auto seconds = endTimer(start);
    
  if (_debug) {
    cout << endl;
    PrintImageSums("[End of outer loop]");
    cout << "[End of outer loop time] " << seconds << endl;
  }

  _reconstructed.Write(_outputName.c_str());

  _reconstructionDone.SetValue();
}

struct coeffInitParameters irtkReconstruction::createCoeffInitParameters() {
  struct coeffInitParameters parameters;
  parameters.debug = _debug;
  parameters.stackFactor = _stackFactor.size();
  parameters.stackIndex = _stackIndex.size();
  parameters.delta = _delta;
  parameters.lambda = _lambda;
  parameters.alpha = _alpha;
  parameters.qualityFactor = _qualityFactor;

  return parameters;
}

struct reconstructionParameters 
irtkReconstruction::CreateReconstructionParameters(int start, int end) {
  struct reconstructionParameters parameters;

  parameters.globalBiasCorrection = _globalBiasCorrection;
  parameters.adaptive = _adaptive;
  parameters.sigmaBias = _sigmaBias;
  parameters.step = _step;
  parameters.sigmaSCPU = _sigmaSCPU;
  parameters.sigmaS2CPU = _sigmaS2CPU;
  parameters.mixSCPU = _mixSCPU;
  parameters.mixCPU = _mixCPU;
  parameters.lowIntensityCutoff = _lowIntensityCutoff;
  parameters.numThreads = _numThreads;
  parameters.start = start;
  parameters.end = end;

  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 3; j++)
      parameters.directions[i][j] = _directions[i][j];

  return parameters;
}

void irtkReconstruction::CoeffInitBootstrap(
    struct coeffInitParameters parameters) {

  cout << "In CoeffInitBootstrap()" << endl;

  auto startTime = startTimer();

  int diff = _slices.size();
  int factor = (int) ceil(diff / (float)(_numBackendNodes));
  int start;
  int end;

  for (int i = 0; i < (int) _numBackendNodes; i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    start = i * factor;
    end = i * factor + factor;
    end = (end > diff) ? diff : end;

    ebbrt::event_manager->SpawnRemote([this, i, index, start, end, parameters]() {

    auto buf = MakeUniqueIOBuf((2 * sizeof(int)) +
        sizeof(struct coeffInitParameters) +
        sizeof(struct reconstructionParameters));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = COEFF_INIT;
    dp.Get<int>() = 1;
    dp.Get<struct coeffInitParameters>() = parameters;

    auto reconstructionParameters = CreateReconstructionParameters(start, end);
    dp.Get<struct reconstructionParameters>() = reconstructionParameters;

    auto sf = std::make_unique<StaticIOBuf>(
        reinterpret_cast<const uint8_t *>(_stackFactor.data()),
        (size_t)(_stackFactor.size() * sizeof(float)));

    auto si = std::make_unique<StaticIOBuf>(
        reinterpret_cast<const uint8_t *>(_stackIndex.data()),
        (size_t)(_stackIndex.size() * sizeof(int)));

    buf->PrependChain(std::move(serializeSlices(start, end, _slices)));
    buf->PrependChain(std::move(serializeImage(_reconstructed)));
    buf->PrependChain(std::move(serializeImage(_mask)));
    buf->PrependChain(std::move(serializeTransformations(_transformations)));
    buf->PrependChain(std::move(sf));
    buf->PrependChain(std::move(si));

    _received = 0;

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[COEFF_INIT].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }
}


void irtkReconstruction::CoeffInit(struct coeffInitParameters parameters) {
  auto start = startTimer();

  cout << "In CoeffInit([params])" << endl;

  for (int i = 0; i < (int) _numBackendNodes; i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, i, parameters, index]() {

    auto buf = MakeUniqueIOBuf((2 * sizeof(int)) + 
        sizeof(struct coeffInitParameters));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = COEFF_INIT; 
    dp.Get<int>() = 0;
    dp.Get<struct coeffInitParameters>() = parameters;

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[COEFF_INIT].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }
}

void irtkReconstruction::CoeffInit(int iteration) {

  cout << "In CoeffInit([iter])" << endl;

  auto parameters = createCoeffInitParameters();
  bool initialize = iteration == 0;

  if (initialize)
    CoeffInitBootstrap(parameters);
  else
    CoeffInit(parameters);
}

void irtkReconstruction::ExcludeSlicesWithOverlap() {
  vector<int> voxelNumTmp;
  for (int i = 0; i < (int) _voxelNum.size(); i++) {
    voxelNumTmp.push_back(_voxelNum[i]);
  }

  //find median
  sort(voxelNumTmp.begin(), voxelNumTmp.end());
  int median = voxelNumTmp[round(voxelNumTmp.size()*0.5)];

  //remember slices with small overlap with ROI
  _smallSlices.clear();
  for (int i = 0; i < (int) _voxelNum.size(); i++)
    if (_voxelNum[i] < 0.1*median)
      _smallSlices.push_back(i);
}

void irtkReconstruction::GaussianReconstruction() {

  auto start = startTimer();

  _voxelNum.resize(_slices.size());
  _reconstructed = 0;
  _volumeWeights.Initialize(_reconstructed.GetImageAttributes());
  _volumeWeights = 0;

  _phase_performance[GAUSSIAN_RECONSTRUCTION].wait +=
      Gather("CoeffInit & GaussianReconstruction");

  _reconstructed /= _volumeWeights;

  ExcludeSlicesWithOverlap();

  auto seconds = endTimer(start);
  _phase_performance[GAUSSIAN_RECONSTRUCTION].time += seconds;

  if (_debug) {
    PrintImageSums("[GaussianReconstruction output]");
    cout << fixed << "[GaussianReconstruction output] _volumeWeights: " 
      << SumImage(_volumeWeights) << endl;
    cout << fixed << "[GaussianReconstruction time] " << seconds << endl;
  }
}

void irtkReconstruction::SimulateSlices(bool initialize) {
  auto start = startTimer();

  cout << "In SimulateSlices()" << endl;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, initialize, i, index]() {

    auto buf = MakeUniqueIOBuf(2*sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = SIMULATE_SLICES;
    dp.Get<int>() = (int) initialize;
    buf->PrependChain(std::move(serializeSlice(_reconstructed)));

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[SIMULATE_SLICES].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }
}

void irtkReconstruction::MStep(int iteration) {

  auto start = startTimer();

  _mSigma = 0.0;
  _mMix = 0.0;
  _mMin = 0.0;
  _mMax = 0.0;
  _mNum = 0.0;

  _phase_performance[M_STEP].wait += Gather("Simulate Slices & MStep");

  if (_mMix > 0) {
    _sigmaCPU = _mSigma / _mMix;
  } else {
    cerr << "ERROR: MStep _mMix <= 0" << endl;
    ebbrt::Cpu::Exit(EXIT_FAILURE);
  }

  if (_sigmaCPU < _step * _step / _sigmaFactor)
    _sigmaCPU = _step * _step / _sigmaFactor;

  if (iteration > 1) 
    _mixCPU = _mMix / _mNum;

  _mCPU = 1 / (_mMax - _mMin);

  auto seconds = endTimer(start);
  _phase_performance[M_STEP].time += seconds;

  if (_debug) {
    cout << "[MStep output] _sigmaCPU: " << _sigmaCPU << endl;
    cout << "[MStep output] _mixCPU: " << _mixCPU << endl;
    cout << "[MStep output] _mCPU: " << _mCPU << endl;
    cout << "[MStep time] " << seconds << endl;
  }
}

void irtkReconstruction::RestoreSliceIntensities() {

  cout << "In RestoreSliceIntensities()" << endl;

  auto start = startTimer();

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = RESTORE_SLICE_INTENSITIES;

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[RESTORE_SLICE_INTENSITIES].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }
}

void irtkReconstruction::ScaleVolume() {

  auto start = startTimer();

  _num = 0;
  _den = 0;

  _phase_performance[SCALE_VOLUME].wait += Gather("ScaleVolume");

  double scale = _num / _den;
  irtkRealPixel *ptr = _reconstructed.GetPointerToVoxels();

  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*ptr>0) *ptr = *ptr * scale;
    ptr++;
  }

  auto seconds = endTimer(start);
  _phase_performance[SCALE_VOLUME].time += seconds;

  if (_debug) {
    PrintImageSums("[ScaleVolume output]");
    cout << "[ScaleVolume time] " << seconds << endl;
  }
}

void irtkReconstruction::SliceToVolumeRegistration() {

   auto start = startTimer();

   for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = SLICE_TO_VOLUME_REGISTRATION;

    buf->PrependChain(std::move(serializeSlice(_reconstructed)));

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[SLICE_TO_VOLUME_REGISTRATION].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  _phase_performance[GAUSSIAN_RECONSTRUCTION].wait +=
      Gather("SliceToVolumeRegistration");

  auto seconds = endTimer(start);
  _phase_performance[GAUSSIAN_RECONSTRUCTION].time += seconds;
  if (_debug) {
    PrintImageSums("[SliceToVolumeRegistration output]");
  }
}

void irtkReconstruction::InitializeRobustStatistics() {

  auto start = startTimer();

  _sigmaSum = 0;
  _numSum = 0;

  _phase_performance[INITIALIZE_ROBUST_STATISTICS].wait +=
      Gather("Simulate Slices & InitializeRobustStatistics");

  _sigmaCPU = _sigmaSum / _numSum;
  _sigmaSCPU = 0.025;
  _mixCPU = 0.9;
  _mixSCPU = 0.9;
  _mCPU = 1 / (2.1 * _maxIntensity - 1.9 * _minIntensity);

  auto seconds = endTimer(start);
  _phase_performance[INITIALIZE_ROBUST_STATISTICS].time += seconds;

  if (_debug) {
    PrintImageSums("[InitializeRobustStatistics output]");
    cout << "[InitializeRobustStatistics output] _sigmaCPU: " 
      << _sigmaCPU << endl;
    cout << "[InitializeRobustStatistics output] _sigmaSCPU: " 
      << _sigmaSCPU << endl;
    cout << "[InitializeRobustStatistics output] _mixCPU: " << _mixCPU << endl;
    cout << "[InitializeRobustStatistics output] _mixSCPU: " 
      << _mixSCPU << endl;
    cout << "[InitializeRobustStatistics output] _mCPU: " << _mCPU << endl;
    cout << "[InitializeRobustStatistics time] " << seconds << endl;
  }
}

void irtkReconstruction::EStepI() {

  auto start = startTimer();

  cout << "In EStepI()" << endl;

  struct eStepParameters parameters;

  parameters.mCPU = _mCPU;
  parameters.sigmaCPU = _sigmaCPU;
  parameters.mixCPU = _mixCPU;

  _sum = 0.0;
  _den = 0.0;
  _den2 = 0.0;
  _sum2 = 0.0;
  _maxs = 0.0;
  _mins = 1.0;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, parameters, i, index]() {

    auto buf = MakeUniqueIOBuf((2 * sizeof(int)) + 
        sizeof(struct eStepParameters));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = E_STEP_I;
    dp.Get<struct eStepParameters>() = parameters; 
    dp.Get<int>() = _smallSlices.size();
	
    auto smallSlicesData = std::make_unique<StaticIOBuf>(
        reinterpret_cast<const uint8_t *>(_smallSlices.data()),
        (size_t)(_smallSlices.size() * sizeof(int)));

    buf->PrependChain(std::move(smallSlicesData));

    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[E_STEP_I].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }
  
  _phase_performance[E_STEP_I].wait += Gather("EStepI");

  if (_den > 0)
    _meanSCPU = _sum / _den;
  else
    _meanSCPU = _mins;

  if (_den2 > 0)
    _meanS2CPU = _sum2 / _den2;
  else
    _meanS2CPU = (_maxs + _meanSCPU) / 2;

  auto seconds = endTimer(start);
  _phase_performance[E_STEP_I].time += seconds;

  if (_debug) {
    cout << "[EStepI output] _sum: " << _sum << endl;
    cout << "[EStepI output] _den: " << _den << endl;
    cout << "[EStepI output] _den2: " << _den2 << endl;
    cout << "[EStepI output] _sum2: " << _sum2 << endl;
    cout << "[EStepI output] _maxs: " << _maxs << endl;
    cout << "[EStepI output] _mins: " << _mins << endl;
    cout << "[EStepI output] _meanSCPU: " << _meanSCPU << endl;
    cout << "[EStepI output] _meanS2CPU: " << _meanS2CPU << endl;
    cout << "[EStepI time] " << seconds << endl;
  }
}

void irtkReconstruction::EStepII() {

  auto start = startTimer();

  cout << "In EStepII()" << endl;

  _sum = 0.0;
  _den = 0.0;
  _den2 = 0.0;
  _sum2 = 0.0;

  struct eStepParameters parameters;
  
  parameters.meanSCPU = _meanSCPU;
  parameters.meanS2CPU = _meanS2CPU;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, parameters, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int) + 
        sizeof(struct eStepParameters));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = E_STEP_II;
    dp.Get<struct eStepParameters>() = parameters; 
	
    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[E_STEP_II].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  _phase_performance[E_STEP_II].wait += Gather("EStepII");

  // [fetalRecontruction] do not allow too small sigma
  if ((_sum > 0) && (_den > 0)) {
    _sigmaSCPU = _sum / _den;
    if (_sigmaSCPU < _step * _step / _sigmaFactor)
      _sigmaSCPU = _step * _step / _sigmaFactor;
  } else {
    _sigmaSCPU = 0.025;
  }

  if ((_sum2 > 0) && (_den2 > 0)) {
    _sigmaS2CPU = _sum2 / _den2;
    if (_sigmaS2CPU < _step * _step / _sigmaFactor)
      _sigmaS2CPU = _step * _step / _sigmaFactor;
  } else {
    _sigmaS2CPU = (_meanS2CPU - _meanSCPU) * (_meanS2CPU - _meanSCPU) / 4;
    if (_sigmaS2CPU < _step * _step / _sigmaFactor)
      _sigmaS2CPU = _step * _step / _sigmaFactor;
  }

  auto seconds = endTimer(start);
  _phase_performance[E_STEP_II].time += seconds;

  if (_debug) {
    cout << "[EStepII output] _sum: " << _sum << endl;
    cout << "[EStepII output] _den: " << _den << endl;
    cout << "[EStepII output] _den2: " << _den2 << endl;
    cout << "[EStepII output] _sum2: " << _sum2 << endl;
    cout << "[EStepII output] _sigmaSCPU: " << _sigmaSCPU << endl;
    cout << "[EStepII output] _sigmaS2CPU: " << _sigmaS2CPU << endl;
    cout << "[EStepII time] " << seconds << endl;
  }
}

void irtkReconstruction::EStepIII() {

  auto start = startTimer();

  cout << "In EStepIII()" << endl;

  _sum = 0.0;
  _num = 0.0;

  struct eStepParameters parameters;
  
  parameters.meanSCPU = _meanSCPU;
  parameters.meanS2CPU = _meanS2CPU;
  parameters.sigmaSCPU = _sigmaSCPU;
  parameters.sigmaS2CPU = _sigmaS2CPU;
  parameters.mixSCPU = _mixSCPU;
  parameters.den = _den;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, parameters, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int) + 
        sizeof(struct eStepParameters));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = E_STEP_III;
    dp.Get<struct eStepParameters>() = parameters; 
	
    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[E_STEP_III].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  _phase_performance[E_STEP_III].wait += Gather("EStepIII");

  if (_num > 0)
    _mixSCPU = _sum / _num;
  else
    _mixSCPU = 0.9;

  auto seconds = endTimer(start);
  _phase_performance[E_STEP_III].time += seconds;

  if (_debug) {
    cout << "[EStepIII output] _sum: " << _sum << endl;
    cout << "[EStepIII output] _num: " << (int) _num << endl;
    cout << "[EStepIII output] _mixSCPU: " << _mixSCPU << endl;
    cout << "[EStepIII time] " << seconds << endl;
  }
}

void irtkReconstruction::EStep() {
  EStepI();

  EStepII();

  EStepIII();
}

void irtkReconstruction::Scale() {
  auto start = startTimer();

  cout << "In Scale()" << endl;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, i, index]() {

    auto buf = MakeUniqueIOBuf(sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = SCALE;
	
    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[SCALE].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  _phase_performance[SCALE].wait += Gather("Scale");

  auto seconds = endTimer(start);
  _phase_performance[SCALE].time += seconds;

  if (_debug) 
    cout << "[Scale time] " << seconds << endl;
}

void irtkReconstruction::AdaptiveRegularization2(vector<irtkRealImage> &_b,
    vector<double> &_factor, irtkRealImage &_original) {

  int dx = _reconstructed.GetX();
  int dy = _reconstructed.GetY();
  int dz = _reconstructed.GetZ();
  for (size_t x = 0; x != (size_t)_reconstructed.GetX(); ++x) {
    int xx, yy, zz;
    for (int y = 0; y < dy; y++)
      for (int z = 0; z < dz; z++) {
        double val = 0;
        double valW = 0;
        double sum = 0;
        for (int i = 0; i < 13; i++) {
          xx = x + _directions[i][0];
          yy = y + _directions[i][1];
          zz = z + _directions[i][2];
          if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
              (zz < dz)) {
            val += _b[i](x, y, z) * _original(xx, yy, zz) *
              _confidenceMap(xx, yy, zz);
            valW += _b[i](x, y, z) * _confidenceMap(xx, yy, zz);
            sum += _b[i](x, y, z);
          }
        }

        for (int i = 0; i < 13; i++) {
          xx = x - _directions[i][0];
          yy = y - _directions[i][1];
          zz = z - _directions[i][2];
          if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
              (zz < dz)) {
            val += _b[i](xx, yy, zz) * _original(xx, yy, zz) *
              _confidenceMap(xx, yy, zz);
            valW += _b[i](xx, yy, zz) * _confidenceMap(xx, yy, zz);
            sum += _b[i](xx, yy, zz);
          }
        }

        val -= sum * _original(x, y, z) * _confidenceMap(x, y, z);
        valW -= sum * _confidenceMap(x, y, z);
        val = _original(x, y, z) * _confidenceMap(x, y, z) +
          _alpha * _lambda / (_delta * _delta) * val;
        valW = _confidenceMap(x, y, z) +
          _alpha * _lambda / (_delta * _delta) * valW;

        if (valW > 0) {
          _reconstructed(x, y, z) = val / valW;
        } else
          _reconstructed(x, y, z) = 0;
      }
  }
}

void irtkReconstruction::AdaptiveRegularization1(vector<irtkRealImage> &_b,
    vector<double> &_factor, irtkRealImage &_original) {
  int dx = _reconstructed.GetX();
  int dy = _reconstructed.GetY();
  int dz = _reconstructed.GetZ();
  for (size_t i = 0; i != 13; ++i) {
    int x, y, z, xx, yy, zz;
    double diff;
    for (x = 0; x < dx; x++)
      for (y = 0; y < dy; y++)
        for (z = 0; z < dz; z++) {
          xx = x + _directions[i][0];
          yy = y + _directions[i][1];
          zz = z + _directions[i][2];
          if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
              (zz < dz) && (_confidenceMap(x, y, z) > 0) &&
              (_confidenceMap(xx, yy, zz) > 0)) {
            diff = (_original(xx, yy, zz) - _original(x, y, z)) *
              sqrt(_factor[i]) / _delta;
            _b[i](x, y, z) = _factor[i] / sqrt(1 + diff * diff);

          } else
            _b[i](x, y, z) = 0;
        }
  }
}

void irtkReconstruction::AdaptiveRegularization(int iteration,
    irtkRealImage &original) {

  vector<double> factor(13, 0);
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 3; j++)
      factor[i] += fabs(double(_directions[i][j]));
    factor[i] = 1 / factor[i];
  }

  vector<irtkRealImage> b; 
  for (int i = 0; i < 13; i++)
    b.push_back(_reconstructed);

  AdaptiveRegularization1(b, factor, original);

  irtkRealImage original2 = _reconstructed;
  AdaptiveRegularization2(b, factor, original2);

  if (_alpha * _lambda / (_delta * _delta) > 0.068) {
    cout << "Warning: regularization might not have smoothing effect! Ensure "
         << "that alpha*lambda/delta^2 is below 0.068." 
         << endl;
  }
}

void irtkReconstruction::BiasCorrectVolume(irtkRealImage &original) {
  // [fetalReconstruction] remove low-frequancy component in the 
  // [fetalReconstruction] reconstructed image which might have accured 
  // [fetalReconstruction] due to overfitting of the biasfield
  irtkRealImage residual = _reconstructed;
  irtkRealImage weights = _mask;

  // [fetalReconstruction] calculate weighted residual
  irtkRealPixel *pr = residual.GetPointerToVoxels();
  irtkRealPixel *po = original.GetPointerToVoxels();
  irtkRealPixel *pw = weights.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    // [fetalReconstruction] second and term to avoid numerical problems
    if ((*pw == 1) && (*po > _lowIntensityCutoff * _maxIntensity) &&
        (*pr > _lowIntensityCutoff * _maxIntensity)) {
      *pr /= *po;
      *pr = log(*pr);
    } else {
      *pw = 0;
      *pr = 0;
    }
    pr++;
    po++;
    pw++;
  }
  // [fetalReconstruction] blurring needs to be same as for slices
  irtkGaussianBlurring<irtkRealPixel> gb(_sigmaBias);
  // [fetalReconstruction] blur weigted residual
  gb.SetInput(&residual);
  gb.SetOutput(&residual);
  gb.Run();
  // [fetalReconstruction] blur weight image
  gb.SetInput(&weights);
  gb.SetOutput(&weights);
  gb.Run();

  // [fetalReconstruction] calculate the bias field
  pr = residual.GetPointerToVoxels();
  pw = weights.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  irtkRealPixel *pi = _reconstructed.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {

    if (*pm == 1) {
      // [fetalReconstruction] weighted gaussian smoothing
      *pr /= *pw;
      // [fetalReconstruction] exponential to recover multiplicative bias field
      *pr = exp(*pr);
      // [fetalReconstruction] bias correct reconstructed
      *pi /= *pr;
      // [fetalReconstruction] clamp intensities to allowed range
      if (*pi < _minIntensity * 0.9)
        *pi = _minIntensity * 0.9;
      if (*pi > _maxIntensity * 1.1)
        *pi = _maxIntensity * 1.1;
    } else {
      *pr = 0;
    }
    pr++;
    pw++;
    pm++;
    pi++;
  }
}

void irtkReconstruction::SuperResolution(int iteration) {

  auto start = startTimer();

  cout << "In SuperResolution()" << endl;

  if (_debug) {
    cout << "[SuperResolution input] iteration: " << iteration << endl;
    cout << "[SuperResolution input] _alpha: " << _alpha << endl;
    cout << "[SuperResolution input] _globalBiasCorrection: " 
      << _globalBiasCorrection << endl;
    cout << "[SuperResolution input] _minIntensity: " << _minIntensity << endl;
    cout << "[SuperResolution input] _maxIntensity: " << _maxIntensity << endl;
  }

  if (iteration == 1) {
    _addon.Initialize(_reconstructed.GetImageAttributes());
    _confidenceMap.Initialize(_reconstructed.GetImageAttributes());
  }
  
  _addon = 0;
  _confidenceMap = 0;
  irtkRealImage original = _reconstructed;

  for (int i = 0; i < (int) _nids.size(); i++) {

    auto index = _frontEnd_cpus_map[_nids[i].ToString()];   // get the cpu index
    auto cpu_i = ebbrt::Cpu::GetByIndex(index);  // get the cpu
    auto ctxt = cpu_i->get_context();  // context

    ebbrt::event_manager->SpawnRemote([this, iteration, i, index]() {

    auto buf = MakeUniqueIOBuf(2*sizeof(int));
    auto dp = buf->GetMutDataPointer();

    dp.Get<int>() = SUPERRESOLUTION;
    dp.Get<int>() = iteration;
	
    cout << "Sending to network: " << _nids[i].ToString();
    cout << " to core: " << index << " data of size: " << buf->ComputeChainDataLength() << endl;
    _phase_performance[SUPERRESOLUTION].sent += buf->ComputeChainDataLength();
    SendMessage(_nids[i], std::move(buf));
    }, ctxt);
  }

  _phase_performance[SUPERRESOLUTION].wait += Gather("SuperResolution");

  if (!_adaptive)
    for (int i = 0; i < _addon.GetX(); i++) {
      for (int j = 0; j < _addon.GetY(); j++) {
        for (int k = 0; k < _addon.GetZ(); k++) {
          if (_confidenceMap(i, j, k) > 0) {
            // [fetalReconstruction] ISSUES if _confidenceMap(i, j, k) is too 
            // [fetalReconstruction] small leading to bright pixels
            _addon(i, j, k) /= _confidenceMap(i, j, k);
            // [fetalReconstruction] this is to revert to normal (non-adaptive) 
            // [fetalReconstruction] regularisation
            _confidenceMap(i, j, k) = 1;
          }
        }
      }
    }

  _reconstructed += _addon * _alpha; 

  // [fetalReconstruction] bound the intensities
  for (int i = 0; i < (int)_reconstructed.GetX(); i++) {
    for (int j = 0; j < (int)_reconstructed.GetY(); j++) {
      for (int k = 0; k < (int)_reconstructed.GetZ(); k++) {
        if (_reconstructed(i, j, k) < _minIntensity * 0.9)
          _reconstructed(i, j, k) = _minIntensity * 0.9;
        if (_reconstructed(i, j, k) > _maxIntensity * 1.1)
          _reconstructed(i, j, k) = _maxIntensity * 1.1;
      }
    }
  }

  // [fetalReconstruction] Smooth the reconstructed image
  // TODO: This can be parallelized.
  AdaptiveRegularization(iteration, original);

  // [fetalReconstruction] Remove the bias in the reconstructed volume 
  // [fetalReconstruction] compared to previous iteration
  // TODO: verify that this works.
  if (_globalBiasCorrection) {
    BiasCorrectVolume(original);
  }

  auto seconds = endTimer(start);
  _phase_performance[SUPERRESOLUTION].time += seconds;

  if (_debug) {
    PrintImageSums("[SuperResolution output]");
    cout << fixed << "[SuperResolution output] _addon: " 
      << SumImage(_addon) << endl;
    cout << fixed << "[SuperResolution output] _confidenceMap: " 
      << SumImage(_confidenceMap) << endl;
    cout << fixed << "[SuperResolution time] " << seconds << endl; 
  }
}

void irtkReconstruction::InitializeEMValues() {
  for (int i = 0; i < _slices.size(); i++) {
    // [fetalRecontruction] Initialize voxel weights and bias values
    irtkRealPixel *pw = _weights[i].GetPointerToVoxels();
    irtkRealPixel *pb = _bias[i].GetPointerToVoxels();
    irtkRealPixel *pi = _slices[i].GetPointerToVoxels();
    for (int j = 0; j < _weights[i].GetNumberOfVoxels(); j++) {
      if (*pi != -1) {
        *pw = 1;
        *pb = 0;
      } else {
        *pw = 0;
        *pb = 0;
      }
      pi++;
      pw++;
      pb++;
    }
    // [fetalRecontruction] Initialize slice weights
    _sliceWeightCPU[i] = 1;
    // [fetalRecontruction] Initialize scaling factors for intensity matching
    _scaleCPU[i] = 1;
  }
}

void irtkReconstruction::ResetOrigin(
    irtkGreyImage &image, irtkRigidTransformation &transformation) {
  double ox, oy, oz;
  image.GetOrigin(ox, oy, oz);
  image.PutOrigin(0, 0, 0);
  transformation.PutTranslationX(ox);
  transformation.PutTranslationY(oy);
  transformation.PutTranslationZ(oz);
  transformation.PutRotationX(0);
  transformation.PutRotationY(0);
  transformation.PutRotationZ(0);
}

void irtkReconstruction::SetSmoothingParameters(double lambda) {
  _lambda = lambda * _delta * _delta;
  _alpha = 0.05 / lambda;
  _alpha = (_alpha > 1) ? 1 : _alpha;
}

/*
 * ReceiveMessage() function
 */
void irtkReconstruction::ReceiveMessage(Messenger::NetworkId nid,
    std::unique_ptr<IOBuf> &&buffer) {
  size_t cpu = ebbrt::Cpu::GetMine();

  cout << "Receiving message from: " << nid.ToString() << " data of size: " << buffer->ComputeChainDataLength();
  cout << " on core: " << cpu << endl;

  auto len = buffer->ComputeChainDataLength();
  auto dp = buffer->GetDataPointer();
  auto fn = dp.Get<int>();

  if (fn < WORK_PHASES)
    _phase_performance[fn].recv += len; 

  switch(fn) {
    case GAUSSIAN_RECONSTRUCTION:
      {
        ReturnFromGaussianReconstruction(dp);
        break;
      }
    case INITIALIZE_ROBUST_STATISTICS:
      {
        ReturnFromInitializeRobustStatistics(dp);
        break;
      }
    case E_STEP_I:
      {
        ReturnFromEStepI(dp);
        break;
      }
    case E_STEP_II:
      {
        ReturnFromEStepII(dp);
        break;
      }
    case E_STEP_III:
      {
        ReturnFromEStepIII(dp);
        break;
      }
    case SCALE:
      {
        ReturnFromScale(dp);
        break;
      }
    case SUPERRESOLUTION:
      {
        ReturnFromSuperResolution(dp);
        break;
      }
    case M_STEP:
      {
        ReturnFromMStep(dp);
        break;
      }
    case SCALE_VOLUME:
      {
        ReturnFromScaleVolume(dp);
        break;
      }
    case SLICE_TO_VOLUME_REGISTRATION:
      {
        ReturnFromSliceToVolumeRegistration(dp);
        break;
      }
    case GATHER_TIMERS:
      {
        ReturnFromGatherTimers(dp);
        break;
      }
    case PING:
      {
          cout << "recevied a ping message" << endl;
          break;
      }
    default:
      {
        cout << "ERROR: ReceiveMessage() invalid option" << endl;
        ebbrt::Cpu::Exit(EXIT_FAILURE);
      }
  } 
}

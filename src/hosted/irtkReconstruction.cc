
#include "irtkReconstruction.h"
#include "../utils.h"

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

void irtkReconstruction::SetDefaultParameters() {
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
      irtkReconstruction::SetDefaultParameters();
}

void irtkReconstruction::SetParameters(arguments args) {
  _outputName = args.outputName;  // Not used
  _maskName = args.maskName; // Not used
  _referenceVolumeName = args.referenceVolumeName; // Not used
  _logId = args.logId; // Not used
  _tFolder = args.tFolder; // Not used
  _sFolder = args.sFolder; // Not used

  _inputStacks = args.inputStacks; // Not used
  _inputTransformations = args.inputTransformations; // Not used
  _thickness = args.thickness; // Not used
  _packages = args.packages; // Not used
  _forceExcluded = args.forceExcluded; // Not used
  _devicesToUse = args.devicesToUse; // Not used

  _iterations = args.iterations;  // Not used
  _levels = args.levels; // Not used
  _recIterationsFirst = args.recIterationsFirst; // Not used
  _recIterationsLast = args.recIterationsLast; // Not used
  _numThreads = args.numThreads; // Not used
  _numBackendNodes = args.numBackendNodes; // Not used
  _numFrontendCPUs = args.numFrontendCPUs; // Not used
  
  _numInputStacksTuner = args.numInputStacksTuner; // Not used
  _T1PackageSize = args.T1PackageSize; // Not used
  _numDevicesToUse = args.numDevicesToUse; // Not used
  
  _sigma = args.sigma; // Not used
  _resolution = args.resolution; // Not used
  _averageValue = args.averageValue; // Not used
  _delta = args.delta; // Not used
  _lambda = args.lambda; // Not used
  _lastIterLambda = args.lastIterLambda; // Not used
  _smoothMask = args.smoothMask; // Not used
  _lowIntensityCutoff = (args.lowIntensityCutoff > 1) ? 1 : 0; // Not used
  
  _globalBiasCorrection = args.globalBiasCorrection; // Not used
  _intensityMatching = args.intensityMatching; // Not used
  _debug = args.debug; // Not used
  _noLog = args.noLog; // Not used
  _useCPU = args.useCPU; // Not used
  _useCPUReg = _useCPU; // Not used
  _useAutoTemplate = args.useAutoTemplate; // Not used
  _useSINCPSF = args.useSINCPSF; // Not used
  _disableBiasCorr = args.disableBiasCorr; // Not used
}

void irtkReconstruction::ReceiveMessage(Messenger::NetworkId nid,
    std::unique_ptr<IOBuf> &&buffer) {

}

void irtkReconstruction::AddNid(ebbrt::Messenger::NetworkId nid) {
  _nids.push_back(nid);
  if ((int) _nids.size() == _numBackendNodes) {
    _backendsAllocated.SetValue();
  }
}

/**
 * Fetal Reconstruction functions
**/ 

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

  // [fetalRecontruction] enlarge stack in z-direction in case top of the head is cut off
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

    // [fetalRecontruction] resample the mask according to the template volume using identity
    // [fetalRecontruction] transformation
    irtkRigidTransformation transformation;
    irtkImageTransformation imagetransformation;
    irtkNearestNeighborInterpolateImageFunction interpolator;
    imagetransformation.SetInput(mask, &transformation);
    imagetransformation.SetOutput(&_mask);
    // [fetalRecontruction] target is zero image, need padding -1
    imagetransformation.PutTargetPaddingValue(-1);
    // [fetalRecontruction] need to fill voxels in target where there is no info from source with
    // [fetalRecontruction] zeroes
    imagetransformation.PutSourcePaddingValue(0);
    imagetransformation.PutInterpolator(&interpolator);
    imagetransformation.Run();
  } else {
    // [fetalRecontruction] fill the mask with ones
    _mask = 1;
  }
  // [fetalRecontruction] set flag that mask was created
  _haveMask = true;

  if (_debug)
    _mask.Write("mask.nii");
}

// TODO: This can be moved into another file
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

// TODO: figure out if this is actually used
/*
irtkRealImage irtkReconstruction::CreateAverage(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations) {

  if (!_templateCreated) {
    cerr << "Please create the template before calculating the average of the "
            "stacks."
         << endl;
    exit(1);
  }

  InvertStackTransformations(stackTransformations);
  ParallelAverage parallelAverage(this, stacks, stackTransformations, -1, 0,
                                  0, // [fetalRecontruction] target/source/background
                                  true, _numThreads);
  parallelAverage();
  irtkRealImage average = parallelAverage.average;
  irtkRealImage weights = parallelAverage.weights;
  average /= weights;
  InvertStackTransformations(stackTransformations);

  return average;
}
*/

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
    if (_debug) {
      sprintf(buffer, "mask-for-matching%i.nii.gz", ind);
      m.Write(buffer);
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
    for (ind = 0; ind < stacks.size(); ind++) {
      sprintf(buffer, "rescaled-stack%i.nii.gz", ind);
      stacks[ind].Write(buffer);
    }

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
        // [fetalRecontruction] if the value is smaller than 1 assume it is padding
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


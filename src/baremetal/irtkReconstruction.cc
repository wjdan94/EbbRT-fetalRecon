#include "irtkReconstruction.h"

static size_t indexToCPU(size_t i) { return i; }
ebbrt::SpinLock spinlock;

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, irtkReconstruction);

irtkReconstruction::irtkReconstruction(EbbId ebbid)
    : Messagable<irtkReconstruction>(ebbid) {}

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

EbbRef<irtkReconstruction> irtkReconstruction::Create(EbbId id) {
  return EbbRef<irtkReconstruction>(id);
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

void printCoeffInitParameters(struct coeffInitParameters parameters) {
  cout << "CoeffInit() Parameters: " << endl;
  cout << "stackFactor: " << parameters.stackFactor << endl;
  cout << "stackIndex: " << parameters.stackIndex << endl;
  cout << "delta: " << parameters.delta << endl;
  cout << "lambda: " << parameters.lambda << endl;
  cout << "alpha: " << parameters.alpha << endl;
  cout << "qualityFactor: " << parameters.qualityFactor << endl;
}

void printReconstructionParameters(struct reconstructionParameters parameters) {
  cout << "Global Bias Correction: " << parameters.globalBiasCorrection << endl;
  cout << "start: " << parameters.start << endl;
  cout << "end: " << parameters.end << endl;
  cout << "Adaptive: " << parameters.adaptive << endl;
  cout << "Sigma Bias: " << parameters.sigmaBias << endl;
  cout << "Step: " << parameters.step << endl;
  cout << "Sigma SCPU: " << parameters.sigmaSCPU << endl;
  cout << "Sigma S2CPU: " << parameters.sigmaS2CPU << endl;
  cout << "Mix SCPU: " << parameters.mixSCPU << endl;
  cout << "Mix CPU: " << parameters.mixCPU << endl;
  cout << "Low Intensity Cutoff" << parameters.lowIntensityCutoff << endl;
}

void irtkReconstruction::StoreParameters(
    struct reconstructionParameters parameters) {
  _globalBiasCorrection = parameters.globalBiasCorrection;
  _adaptive = parameters.adaptive;
  _sigmaBias = parameters.sigmaBias;
  _step = parameters.step;
  _sigmaSCPU = parameters.sigmaSCPU;
  _sigmaS2CPU = parameters.sigmaS2CPU;
  _mixSCPU = parameters.mixSCPU;
  _mixCPU = parameters.mixCPU;
  _lowIntensityCutoff = parameters.lowIntensityCutoff;
  _numThreads = parameters.numThreads;
  _start = parameters.start;
  _end = parameters.end;

  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 3; j++)
      _directions[i][j] = parameters.directions[i][j];
}

struct coeffInitParameters irtkReconstruction::StoreCoeffInitParameters(
    ebbrt::IOBuf::DataPointer& dp) {
  auto parameters = dp.Get<struct coeffInitParameters>();
  auto reconstructionParameters = dp.Get<struct reconstructionParameters>();

  StoreParameters(reconstructionParameters);
  
  // TODO: delete prints
  //printCoeffInitParameters(parameters);
  //printReconstructionParameters(reconstructionParameters);

  _debug = parameters.debug;
  int stackFactorSize = parameters.stackFactor;
  int stackIndexSize = parameters.stackIndex;
  _delta = parameters.delta;
  _lambda = parameters.lambda;
  _alpha = parameters.lambda;
  _qualityFactor = parameters.qualityFactor;

  auto nSlices = dp.Get<int>();
  _slices.resize(nSlices);

  DeserializeSliceVector(dp, nSlices);
  DeserializeSlice(dp, _reconstructed);
  DeserializeSlice(dp, _mask);

  auto nRigidTrans = dp.Get<int>();	

  _transformations.resize(nRigidTrans);
      
  for(int i = 0; i < nRigidTrans; i++) {
    DeserializeTransformations(dp, _transformations[i]);
  }

  _stackFactor.resize(stackFactorSize);
  dp.Get(stackFactorSize*sizeof(float), (uint8_t*)_stackFactor.data());
      
  _stackIndex.resize(stackIndexSize);
  dp.Get(stackIndexSize*sizeof(int), (uint8_t*)_stackIndex.data());

  return parameters;
}

void irtkReconstruction::InitializeEMValues() {
  for (int i = 0; i < (int) _slices.size(); i++) {
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

void irtkReconstruction::InitializeEM() {
  _weights.clear();
  _bias.clear();
  _scaleCPU.clear();
  _sliceWeightCPU.clear();
  _slicePotential.clear();

  for (int i = 0; i < (int) _slices.size(); i++) {
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

void irtkReconstruction::ParallelCoeffInit(int start, int end) {
  for (size_t index = start; (int) index != end; ++index) {

    bool sliceInside;

    //get resolution of the volume
    double vx, vy, vz;
    _reconstructed.GetPixelSize(&vx, &vy, &vz);
    //volume is always isotropic
    double res = vx;

    //start of a loop for a slice index
    cout << index << " ";

    //read the slice
    irtkRealImage& slice = _slices[index];

    //prepare structures for storage
    POINT3D p;
    VOXELCOEFFS empty;
    SLICECOEFFS slicecoeffs(slice.GetX(), vector < VOXELCOEFFS >(slice.GetY(), empty));

    //to check whether the slice has an overlap with mask ROI
    sliceInside = false;

    //PSF will be calculated in slice space in higher resolution

    //get slice voxel size to define PSF
    double dx, dy, dz;
    slice.GetPixelSize(&dx, &dy, &dz);

    //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
    double sigmax = 1.2 * dx / 2.3548;
    double sigmay = 1.2 * dy / 2.3548;
    double sigmaz = dz / 2.3548;

    //calculate discretized PSF
    //isotropic voxel size of PSF - derived from resolution of reconstructed volume
    double size = res / _qualityFactor;

    //number of voxels in each direction
    //the ROI is 2*voxel dimension

    int xDim = round(2 * dx / size);
    int yDim = round(2 * dy / size);
    int zDim = round(2 * dz / size);

    //image corresponding to PSF
    irtkImageAttributes attr;
    attr._x = xDim;
    attr._y = yDim;
    attr._z = zDim;
    attr._dx = size;
    attr._dy = size;
    attr._dz = size;
    irtkRealImage PSF(attr);

    //centre of PSF
    double cx, cy, cz;
    cx = 0.5 * (xDim - 1);
    cy = 0.5 * (yDim - 1);
    cz = 0.5 * (zDim - 1);
    PSF.ImageToWorld(cx, cy, cz);

    double x, y, z;
    double sum = 0;
    int i, j, k;
    for (i = 0; i < xDim; i++)
      for (j = 0; j < yDim; j++)
        for (k = 0; k < zDim; k++) {
          x = i;
          y = j;
          z = k;
          PSF.ImageToWorld(x, y, z);
          x -= cx;
          y -= cy;
          z -= cz;
          //continuous PSF does not need to be normalized as discrete will be
          PSF(i, j, k) = exp(
              -x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay)
              - z * z / (2 * sigmaz * sigmaz));
          sum += PSF(i, j, k);
        }
    PSF /= sum;

    //prepare storage for PSF transformed and resampled to the space of reconstructed volume
    //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
    //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
    int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2))
      * 2 + 1 + 2;
    //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
    attr._x = dim;
    attr._y = dim;
    attr._z = dim;
    attr._dx = res;
    attr._dy = res;
    attr._dz = res;
    //create matrix from transformed PSF
    irtkRealImage tPSF(attr);
    //calculate centre of tPSF in image coordinates
    int centre = (dim - 1) / 2;

    //for each voxel in current slice calculate matrix coefficients
    int ii, jj, kk;
    int tx, ty, tz;
    int nx, ny, nz;
    int l, m, n;
    double weight;
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
          x = i;
          y = j;
          z = 0;
          slice.ImageToWorld(x, y, z);
          _transformations[index].Transform(x, y, z);
          _reconstructed.WorldToImage(x, y, z);
          tx = round(x);
          ty = round(y);
          tz = round(z);

          //Clear the transformed PSF
          for (ii = 0; ii < dim; ii++)
            for (jj = 0; jj < dim; jj++)
              for (kk = 0; kk < dim; kk++)
                tPSF(ii, jj, kk) = 0;

          //for each POINT3D of the PSF
          for (ii = 0; ii < xDim; ii++)
            for (jj = 0; jj < yDim; jj++)
              for (kk = 0; kk < zDim; kk++) {
                //Calculate the position of the POINT3D of
                //PSF centered over current slice voxel                            
                //This is a bit complicated because slices
                //can be oriented in any direction 

                //PSF image coordinates
                x = ii;
                y = jj;
                z = kk;
                //change to PSF world coordinates - now real sizes in mm
                PSF.ImageToWorld(x, y, z);
                //centre around the centrepoint of the PSF
                x -= cx;
                y -= cy;
                z -= cz;

                //Need to convert (x,y,z) to slice image
                //coordinates because slices can have
                //transformations included in them (they are
                //nifti)  and those are not reflected in
                //PSF. In slice image coordinates we are
                //sure that z is through-plane 

                //adjust according to voxel size
                x /= dx;
                y /= dy;
                z /= dz;
                //center over current voxel
                x += i;
                y += j;

                //convert from slice image coordinates to world coordinates
                slice.ImageToWorld(x, y, z);

                //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                //Transform to space of reconstructed volume
                _transformations[index].Transform(x, y, z);
                //Change to image coordinates
                _reconstructed.WorldToImage(x, y, z);

                //determine coefficients of volume voxels for position x,y,z
                //using linear interpolation

                //Find the 8 closest volume voxels

                //lowest corner of the cube
                nx = (int)floor(x);
                ny = (int)floor(y);
                nz = (int)floor(z);

                //not all neighbours might be in ROI, thus we need to normalize
                //(l,m,n) are image coordinates of 8 neighbours in volume space
                //for each we check whether it is in volume
                sum = 0;
                //to find wether the current slice voxel has overlap with ROI
                bool inside = false;
                for (l = nx; l <= nx + 1; l++)
                  if ((l >= 0) && (l < _reconstructed.GetX()))
                    for (m = ny; m <= ny + 1; m++)
                      if ((m >= 0) && (m < _reconstructed.GetY()))
                        for (n = nz; n <= nz + 1; n++)
                          if ((n >= 0) && (n < _reconstructed.GetZ())) {
                            weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                            sum += weight;
                            if (_mask(l, m, n) == 1) {
                              inside = true;
                              sliceInside = true;
                            }
                          }
                //if there were no voxels do nothing
                if ((sum <= 0) || (!inside))
                  continue;
                //now calculate the transformed PSF
                for (l = nx; l <= nx + 1; l++)
                  if ((l >= 0) && (l < _reconstructed.GetX()))
                    for (m = ny; m <= ny + 1; m++)
                      if ((m >= 0) && (m < _reconstructed.GetY()))
                        for (n = nz; n <= nz + 1; n++)
                          if ((n >= 0) && (n < _reconstructed.GetZ())) {
                            weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                            //image coordinates in tPSF
                            //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                            int aa, bb, cc;
                            aa = l - tx + centre;
                            bb = m - ty + centre;
                            cc = n - tz + centre;

                            //resulting value
                            double value = PSF(ii, jj, kk) * weight / sum;

                            //Check that we are in tPSF
                            if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0)
                                || (cc >= dim)) {
                              cerr << "Error while trying to populate tPSF. " << aa << " " << bb
                                << " " << cc << endl;
                              cerr << l << " " << m << " " << n << endl;
                              cerr << tx << " " << ty << " " << tz << endl;
                              cerr << centre << endl;
                              exit(1);
                            }
                            else
                              //update transformed PSF
                              tPSF(aa, bb, cc) += value;
                          }
              } 

          //store tPSF values
          for (ii = 0; ii < dim; ii++)
            for (jj = 0; jj < dim; jj++)
              for (kk = 0; kk < dim; kk++)
                if (tPSF(ii, jj, kk) > 0) {
                  p.x = ii + tx - centre;
                  p.y = jj + ty - centre;
                  p.z = kk + tz - centre;
                  p.value = tPSF(ii, jj, kk);
                  slicecoeffs[i][j].push_back(p);
                }
        } //end of loop for slice voxels

    _volcoeffs[index] = slicecoeffs;
    _sliceInsideCPU[index] = sliceInside;
  }  
}

void irtkReconstruction::CoeffInit(ebbrt::IOBuf::DataPointer& dp) {

  auto parameters = StoreCoeffInitParameters(dp);
  
  InitializeEM();
  InitializeEMValues();
  
  _volcoeffs.clear();
  _volcoeffs.resize(_slices.size());

  _sliceInsideCPU.clear();
  _sliceInsideCPU.resize(_slices.size());

  int diff = _end - _start;

  //cout << "nCPUs: " << nCPUs << endl;
  //cout << "numThreas: " << _numThreads << endl;

	//int factor = (int)ceil(diff / (float)_numThreads);
	int factor = (int)ceil(diff / (float)1);
  int start = 0 * factor;
  int end = 0 * factor + factor;
  end = (end > diff) ? diff : end;
  /*

  size_t nCPUs = ebbrt::Cpu::Count();
  static ebbrt::SpinBarrier bar(nCPUs);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t mainCPU = ebbrt::Cpu::GetMine();
  
  cout << "nCPUs: " << nCPUs << endl;
  for (size_t i = 0; i <  nCPUs; i++) {
    cout << "indexToCPU(" << i << "): " << indexToCPU(i)
    ebbrt::event_manager->SpawnRemote([nCPUs, &count, mainCPU, &context]{
        size_t cpu = ebbrt::Cpu::GetMine();

        cout << "CPU: " << cpu << endl;
	      
        {
          std::lock_guard<ebbrt::SpinLock> l(spinlock);
        }
        
        count++;

	      bar.Wait();
        while (count < (size_t)nCPUs)
        ;


        if (cpu == mainCPU) 
          ebbrt::event_manager->ActivateContext(std::move(context));

        }, indexToCPU(i));
  }
  
  ebbrt::event_manager->SaveContext(context);
 */
  ParallelCoeffInit(start, end);

  _volumeWeights.Initialize(_reconstructed.GetImageAttributes());
  _volumeWeights = 0;

  int i, j, n, k, inputIndex;
  POINT3D p;
  for (inputIndex = 0; inputIndex < (int) _slices.size(); ++inputIndex) {
    for (i = 0; i < _slices[inputIndex].GetX(); i++) {
      for (j = 0; j < _slices[inputIndex].GetY(); j++) {
        n = _volcoeffs[inputIndex][i][j].size();
        for (k = 0; k < n; k++) {
          p = _volcoeffs[inputIndex][i][j][k];
          _volumeWeights(p.x, p.y, p.z) += p.value;
        }
      }
    }
  }

  // find average volume weight to modify alpha parameters accordingly
  irtkRealPixel *ptr = _volumeWeights.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  double sum = 0;
  int num = 0;
  for (int i = 0; i < _volumeWeights.GetNumberOfVoxels(); i++) {
    if (*pm == 1) {
      sum += *ptr;
      num++;
    }
    ptr++;
    pm++;
  }
  _averageVolumeWeight = sum / num;


  if (_debug) {
    cout << fixed << "_averageVolumeWeight: " << _averageVolumeWeight << endl;
    cout << fixed << "_volumeWeights: " << SumImage(_volumeWeights) << endl;
  }
}

void irtkReconstruction::GaussianReconstruction() {
  unsigned int inputIndex;
  int i, j, k, n;
  irtkRealImage slice;
  double scale;
  POINT3D p;
  vector<int> voxelNum;
  int sliceVoxNum;

  voxelNum.resize(_end - _start);

  //clear _reconstructed image
  _reconstructed = 0;

  for (inputIndex = _start; (int) inputIndex < _end; ++inputIndex) {
    slice = _slices[inputIndex];
    irtkRealImage& b = _bias[inputIndex];
    scale = _scaleCPU[inputIndex];
    sliceVoxNum = 0;

    //Distribute slice intensities to the volume
    for (i = 0; i < slice.GetX(); i++) {
      for (j = 0; j < slice.GetY(); j++) {
        if (slice(i, j, 0) != -1) {
          //biascorrect and scale the slice
          slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

          //number of volume voxels with non-zero coefficients
          //for current slice voxel
          n = _volcoeffs[inputIndex][i][j].size();

          //if given voxel is not present in reconstructed volume at all,
          //pad it
          if (n > 0)
            sliceVoxNum++;

          //add contribution of current slice voxel to all voxel volumes
          //to which it contributes
          for (k = 0; k < n; k++) {
            p = _volcoeffs[inputIndex][i][j][k];
            _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
          }
        }
      }
    }
    voxelNum[inputIndex - _start] = sliceVoxNum;
  }

  //normalize the volume by proportion of contributing slice voxels
  //for each volume voxe
  _reconstructed /= _volumeWeights;


  //TODO: Verify if this must be done at the frontend level
  vector<int> voxelNumTmp;
  for (i = 0; i < (int) voxelNum.size(); i++)
    voxelNumTmp.push_back(voxelNum[i]);

  //find median
  sort(voxelNumTmp.begin(), voxelNumTmp.end());
  int median = voxelNumTmp[round(voxelNumTmp.size()*0.5)];

  //remember slices with small overlap with ROI
  _smallSlices.clear();
  for (i = 0; i < (int) voxelNum.size(); i++)
    if (voxelNum[i] < 0.1*median)
      _smallSlices.push_back(i);
}

void irtkReconstruction::ParallelSimulateSlices(int start, int end) {
  for (int inputIndex = start; inputIndex != end; ++inputIndex) {
    _simulatedSlices[inputIndex].Initialize(
        _slices[inputIndex].GetImageAttributes());

    _simulatedSlices[inputIndex] = 0;

    _simulatedWeights[inputIndex].Initialize(
        _slices[inputIndex].GetImageAttributes());

    _simulatedWeights[inputIndex] = 0;

    _simulatedInside[inputIndex].Initialize(
        _slices[inputIndex].GetImageAttributes());
    
    _simulatedInside[inputIndex] = 0;
    _sliceInsideCPU[inputIndex] = 0;

    POINT3D p;
    for (unsigned int i = 0; (int) i < _slices[inputIndex].GetX();
         i++) {
      for (unsigned int j = 0; (int) j < _slices[inputIndex].GetY();
           j++) {
        if (_slices[inputIndex](i, j, 0) != -1) {
          double weight = 0;
          int n = _volcoeffs[inputIndex][i][j].size();

          for (unsigned int k = 0; (int) k < n; k++) {
            p = _volcoeffs[inputIndex][i][j][k];

            _simulatedSlices[inputIndex](i, j, 0) +=
                p.value * _reconstructed(p.x, p.y, p.z);
            weight += p.value;

            if (_mask(p.x, p.y, p.z) == 1) {
              _simulatedInside[inputIndex](i, j, 0) = 1;
              _sliceInsideCPU[inputIndex] = 1;
            }
          }

          if (weight > 0) {
            _simulatedSlices[inputIndex](i, j, 0) /= weight;
            _simulatedWeights[inputIndex](i, j, 0) = weight;
          }
        }
      }
    }
  }
}

void irtkReconstruction::SimulateSlices() {
  _simulatedSlices.resize(_end - _start);
  _simulatedInside.resize(_end - _start);
  _simulatedWeights.resize(_end - _start);
  
  ParallelSimulateSlices(_start, _end);
}


void irtkReconstruction::DeserializeTransformations(
    ebbrt::IOBuf::DataPointer& dp, irtkRigidTransformation& tmp) {
  auto tx = dp.Get<double>();
  auto ty = dp.Get<double>();
  auto tz = dp.Get<double>();

  auto rx = dp.Get<double>();
  auto ry = dp.Get<double>();
  auto rz = dp.Get<double>();

  auto cosrx = dp.Get<double>();
  auto cosry = dp.Get<double>();
  auto cosrz = dp.Get<double>();

  auto sinx = dp.Get<double>();
  auto siny = dp.Get<double>();
  auto sinz = dp.Get<double>();

  auto status0 = dp.Get<int>();
  auto status1 = dp.Get<int>();
  auto status2 = dp.Get<int>();
  auto status3 = dp.Get<int>();
  auto status4 = dp.Get<int>();
  auto status5 = dp.Get<int>();

  auto rows = dp.Get<int>();
  auto cols = dp.Get<int>();
  auto ptr = std::make_unique<double[]>(rows * cols);
  dp.Get(rows * cols * sizeof(double), (uint8_t*)ptr.get());
  irtkMatrix mat(rows, cols, std::move(ptr));

  irtkRigidTransformation irt(tx, ty, tz, rx, ry, rz, cosrx, cosry, cosrz, 
      sinx, siny, sinz, status0, status1, status2, status3, status4, 
      status5, mat);

  tmp = std::move(irt);
}

void irtkReconstruction::DeserializeSliceVector(ebbrt::IOBuf::DataPointer& dp,
    int nSlices) {
  for (int i = 0; i < nSlices; i++) {
    DeserializeSlice(dp, _slices[i]);
  }
}

void irtkReconstruction::ReturnFrom(int fn, Messenger::NetworkId frontEndNid) {
  auto buf = MakeUniqueIOBuf(sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = fn;
  SendMessage(frontEndNid, std::move(buf));
}


void irtkReconstruction::DeserializeSlice(ebbrt::IOBuf::DataPointer& dp, 
    irtkRealImage& tmp) {
  auto x = dp.Get<int>();
  auto y = dp.Get<int>();
  auto z = dp.Get<int>();
  auto t = dp.Get<int>();

  auto dx = dp.Get<double>();
  auto dy = dp.Get<double>();
  auto dz = dp.Get<double>();
  auto dt = dp.Get<double>();

  auto xorg = dp.Get<double>();
  auto yorg = dp.Get<double>();
  auto zorg = dp.Get<double>();
  auto torg = dp.Get<double>();

  auto xa0 = dp.Get<double>();
  auto xa1 = dp.Get<double>();
  auto xa2 = dp.Get<double>();

  auto ya0 = dp.Get<double>();
  auto ya1 = dp.Get<double>();
  auto ya2 = dp.Get<double>();

  auto za0 = dp.Get<double>();
  auto za1 = dp.Get<double>();
  auto za2 = dp.Get<double>();

  irtkImageAttributes at(
      x, y, z, t,
      dx, dy, dz, dt,
      xorg, yorg, zorg, torg,
      xa0, xa1, xa2, ya0,
      ya1, ya2, za0, za1,
      za2
      );

  auto rows = dp.Get<int>();
  auto cols = dp.Get<int>();
  //auto ptr = new double[rows * cols];
  auto ptr = std::make_unique<double[]>(rows * cols);
  dp.Get(rows * cols * sizeof(double), (uint8_t*)ptr.get());
  irtkMatrix matI2W(rows, cols, std::move(ptr));

  rows = dp.Get<int>();
  cols = dp.Get<int>();
  //ptr = new double[rows * cols];
  ptr = std::make_unique<double[]>(rows * cols);
  dp.Get(rows * cols * sizeof(double), (uint8_t*)ptr.get());
  irtkMatrix matW2I(rows, cols, std::move(ptr));

  auto n = dp.Get<int>();
  auto ptr2 = new double[n];
  dp.Get(n*sizeof(double), (uint8_t*)ptr2);

  irtkRealImage ri(at, ptr2, matI2W, matW2I);

  tmp = std::move(ri);
}

void irtkReconstruction::ReceiveMessage(Messenger::NetworkId nid,
    std::unique_ptr<IOBuf> &&buffer) {
  auto dp = buffer->GetDataPointer();
  auto fn = dp.Get<int>();

  cout << "Receiving function " << fn << endl;

  switch(fn) {
    case COEFF_INIT:
      {
        CoeffInit(dp);
        //TODO: delete prints
        cout << "-----------------------------" << endl;
        cout << "After COEFF_INIT" << endl;
        PrintImageSums(); 
        PrintAttributeVectorSums();
        for (int i = 0; i < (int) _transformations.size(); i++)
          _transformations[i].PrintTransformation();
        ReturnFrom(COEFF_INIT, nid);
        break;
      }
    case GAUSSIAN_RECONSTRUCTION:
      {
        GaussianReconstruction();
        //TODO: delete prints
        cout << "-----------------------------" << endl;
        cout << "After GAUSSIAN_RECONSTRUCTION" << endl;
        PrintImageSums(); 
        PrintAttributeVectorSums();
        ReturnFrom(GAUSSIAN_RECONSTRUCTION, nid);
        for (int i = 0; i < (int) _transformations.size(); i++)
          _transformations[i].PrintTransformation();
        break;
      }
    case SIMULATE_SLICES:
      {
        SimulateSlices();
        cout << "-----------------------------" << endl;
        cout << "After SIMULATE_SLICES" << endl;
        PrintImageSums(); 
        PrintAttributeVectorSums();
        ReturnFrom(SIMULATE_SLICES, nid);
        for (int i = 0; i < (int) _transformations.size(); i++)
          _transformations[i].PrintTransformation();

        PrintVectorSums(_simulatedSlices, "Simulated Slices");
        PrintVectorSums(_simulatedWeights, "Simulated Weights");
        PrintVectorSums(_simulatedInside, "Simulated Inside");

        cout << "Reconstructed: " << SumImage(_reconstructed) << endl;

        break;
      }
    default:
      cout << "Invalid option" << endl;
  }
}

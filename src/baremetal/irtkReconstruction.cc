// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "irtkReconstruction.h"
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>

#pragma GCC diagnostic ignored "-Wsign-compare"

ebbrt::SpinLock spinLock;

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

void irtkReconstruction::Ping(Messenger::NetworkId nid) {
  // Construct and send the ping message
  auto buf = MakeUniqueIOBuf(sizeof(uint32_t));
  auto dp = buf->GetMutDataPointer();

  dp.Get<int>() = PING;
  SendMessage(nid, std::move(buf));
  std::printf("Ping SetMessage\n");
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

void irtkReconstruction::ResetOrigin(
    irtkRealImage &image, irtkRigidTransformation &transformation) {
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

/*
 * CoeffInit functions
 */

void printCoeffInitParameters(struct coeffInitParameters parameters) {
  cout << "[CoeffInit input] stackFactor: " << parameters.stackFactor << endl;
  cout << "[CoeffInit input] stackIndex: " << parameters.stackIndex << endl;
  cout << "[CoeffInit input] delta: " << parameters.delta << endl;
  cout << "[CoeffInit input] lambda: " << parameters.lambda << endl;
  cout << "[CoeffInit input] alpha: " << parameters.alpha << endl;
  cout << "[CoeffInit input] qualityFactor: " << parameters.qualityFactor << endl;
}

void printReconstructionParameters(struct reconstructionParameters parameters) {
  cout << "[CoeffInit input] Global Bias Correction: " << parameters.globalBiasCorrection << endl;
  cout << "[CoeffInit input] start: " << parameters.start << endl;
  cout << "[CoeffInit input] end: " << parameters.end << endl;
  cout << "[CoeffInit input] Adaptive: " << parameters.adaptive << endl;
  cout << "[CoeffInit input] Sigma Bias: " << parameters.sigmaBias << endl;
  cout << "[CoeffInit input] Step: " << parameters.step << endl;
  cout << "[CoeffInit input] Sigma SCPU: " << parameters.sigmaSCPU << endl;
  cout << "[CoeffInit input] Sigma S2CPU: " << parameters.sigmaS2CPU << endl;
  cout << "[CoeffInit input] Mix SCPU: " << parameters.mixSCPU << endl;
  cout << "[CoeffInit input] Mix CPU: " << parameters.mixCPU << endl;
  cout << "[CoeffInit input] Low Intensity Cutoff" << parameters.lowIntensityCutoff << endl;
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
  // Subtract 1 from _numThreads for the cpu reserved from IO
  _factor = (int) ceil((_end - _start) / (float) (_numThreads - 1));

  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 3; j++)
      _directions[i][j] = parameters.directions[i][j];
}

void irtkReconstruction::StoreCoeffInitParameters(
    ebbrt::IOBuf::DataPointer& dp) {
  
  auto parameters = dp.Get<struct coeffInitParameters>();
  if (_debug)
    printCoeffInitParameters(parameters);
  
  _delta = parameters.delta;
  _lambda = parameters.lambda;
  _qualityFactor = parameters.qualityFactor;
}

void irtkReconstruction::DefineWorkers() {
  for (size_t worker = 0; worker < _numThreads; worker++) {
    if (worker == _IOCPU) {
      if (_debug) 
        cout << "Core #" << worker << " reserved for IO" << endl;
      continue;
    }

    _workers.push_back(worker);
    if (_debug) 
      cout << "Core #" << worker << " added to the pool of workers" << endl;
  }
}

void irtkReconstruction::CoeffInitBootstrap(ebbrt::IOBuf::DataPointer& dp, 
    size_t cpu) {

  cout << "In CoeffInitBootstrap() with IO_CPU " << cpu << endl;

  auto parameters = dp.Get<struct coeffInitParameters>();
  auto reconstructionParameters = dp.Get<struct reconstructionParameters>();

  _debug = parameters.debug;
  _IOCPU = cpu;

  if (_debug) {
    printCoeffInitParameters(parameters);
    printReconstructionParameters(reconstructionParameters);
  }

  _delta = parameters.delta;
  _lambda = parameters.lambda;
  _qualityFactor = parameters.qualityFactor;

  StoreParameters(reconstructionParameters);

  DefineWorkers();

  int stackFactorSize = parameters.stackFactor;
  int stackIndexSize = parameters.stackIndex;

  auto nSlices = dp.Get<int>();
  _slices.resize(nSlices);

  for (int i = _start; i < _end; i++) {
    deserializeSlice(dp, _slices[i]);
  }

  deserializeSlice(dp, _reconstructed);
  deserializeSlice(dp, _mask);

  auto nRigidTrans = dp.Get<int>();	
  _transformations.resize(nRigidTrans);
  for(int i = 0; i < nRigidTrans; i++) {
    deserializeTransformations(dp, _transformations[i]);
  }

  _stackFactor.resize(stackFactorSize);
  dp.Get(stackFactorSize*sizeof(float), (uint8_t*)_stackFactor.data());

  _stackIndex.resize(stackIndexSize);
  dp.Get(stackIndexSize*sizeof(int), (uint8_t*)_stackIndex.data());
  
  InitializeEM();
  
  _voxelNum.resize(_slices.size());
}

void irtkReconstruction::InitializeEMValues() {
  for (int i = _start; i < _end; i++) {
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

  _weights.resize(_slices.size());
  _bias.resize(_slices.size());
  _scaleCPU.resize(_slices.size());
  _sliceWeightCPU.resize(_slices.size());
  _slicePotential.resize(_slices.size());

  for (int i = _start; i < _end; i++) {
    // [fetalRecontruction] Create images for voxel weights and bias fields
    _weights[i] = _slices[i];
    _bias[i] = _slices[i];
    // [fetalRecontruction] Create and initialize scales
    _scaleCPU[i] = 1;
    // [fetalRecontruction] Create and initialize slice weights
    _sliceWeightCPU[i] = 1;
    _slicePotential[i] = 0;
  }

  // [fetalRecontruction] Find the range of intensities
  _maxIntensity = voxel_limits<irtkRealPixel>::min();
  _minIntensity = voxel_limits<irtkRealPixel>::max();
  for (unsigned int i = _start; i < _end; i++) {
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

void irtkReconstruction::ParallelCoeffInit() {
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {

    auto workerId = _workers.at(workerIndex);

    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex]() {

      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      for (size_t index = start; (int) index < end; ++index) {

        bool sliceInside;

        //get resolution of the volume
        double vx, vy, vz;
        _reconstructed.GetPixelSize(&vx, &vy, &vz);
        //volume is always isotropic
        double res = vx;

        //read the slice
        irtkRealImage& slice = _slices[index];

        //prepare structures for storage
        POINT3D p;
        VOXELCOEFFS empty;
        SLICECOEFFS slicecoeffs(slice.GetX(),
            vector < VOXELCOEFFS >(slice.GetY(), empty));

        //to check whether the slice has an overlap with mask ROI
        sliceInside = false;

        //PSF will be calculated in slice space in higher resolution

        //get slice voxel size to define PSF
        double dx, dy, dz;
        slice.GetPixelSize(&dx, &dy, &dz);

        //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, 
        //Gaussian with FWHM = dz through-plane)
        double sigmax = 1.2 * dx / 2.3548;
        double sigmay = 1.2 * dy / 2.3548;
        double sigmaz = dz / 2.3548;

        //calculate discretized PSF
        //isotropic voxel size of PSF - derived from resolution of 
        //reconstructed volume
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

        //prepare storage for PSF transformed and resampled to the space of
        //reconstructed volume maximum dim of rotated kernel - the next higher odd
        //integer plus two to accound for rounding error of tx,ty,tz.  Note
        //conversion from PSF image coordinates to tPSF image coordinates *size/res

        int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) 
                * size / res) / 2)) * 2 + 1 + 2;
        //prepare image attributes. Voxel dimension will be taken from the
        //reconstructed volume
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
                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) 
                                  * (1 - fabs(n - z));
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
                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) 
                                  * (1 - fabs(n - z));

                                //image coordinates in tPSF
                                //(centre,centre,centre) in tPSF is aligned with
                                //(tx,ty,tz)
                                int aa, bb, cc;
                                aa = l - tx + centre;
                                bb = m - ty + centre;
                                cc = n - tz + centre;

                                //resulting value
                                double value = PSF(ii, jj, kk) * weight / sum;

                                //Check that we are in tPSF
                                if ((aa < 0) || (aa >= dim) || (bb < 0) 
                                    || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                  cerr << "Error while trying to populate tPSF. " 
                                    << aa << " " << bb
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
      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::CoeffInit(ebbrt::IOBuf::DataPointer& dp, 
    size_t cpu) {

  bool initialize = dp.Get<int>();

  if (initialize)
    CoeffInitBootstrap(dp, cpu);
  else
    StoreCoeffInitParameters(dp);
  
  InitializeEMValues();

  _volcoeffs.clear();
  _volcoeffs.resize(_slices.size());

  _sliceInsideCPU.clear();
  _sliceInsideCPU.resize(_slices.size());

  ParallelCoeffInit();

  _volumeWeights.Initialize(_reconstructed.GetImageAttributes());
  _volumeWeights = 0;

  int i, j, n, k, inputIndex;
  POINT3D p;
  for (inputIndex = _start; inputIndex < (int) _end; ++inputIndex) {
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
}
/* End of CoeffInit functions */

/*
 * GaussianReconstruction functions
 */
void irtkReconstruction::GaussianReconstruction() {
  int inputIndex;
  int i, j, k, n;
  irtkRealImage slice;
  double scale;
  POINT3D p;
  int sliceVoxNum;

  //clear _reconstructed image
  _reconstructed = 0;

  for (inputIndex = _start; inputIndex < _end; ++inputIndex) {
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

    _voxelNum[inputIndex] = sliceVoxNum;
  }
}

void irtkReconstruction::ReturnFromGaussianReconstruction(
    Messenger::NetworkId frontEndNid) {

  cout << "In ReturnFromGaussianReconstruction() to send back to " << frontEndNid.ToString() << " from IO Core: " << _IOCPU << endl;

  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid]() {
 
      auto buf = MakeUniqueIOBuf(3 * sizeof(int));
      auto dp = buf->GetMutDataPointer();

      dp.Get<int>() = GAUSSIAN_RECONSTRUCTION;
      dp.Get<int>() = _start;
      dp.Get<int>() = _end;

      auto vnum = std::make_unique<StaticIOBuf>(
        reinterpret_cast<const uint8_t *>(_voxelNum.data() + _start),
        (size_t)((_end-_start) * sizeof(int)));

      buf->PrependChain(std::move(vnum));
      buf->PrependChain(std::move(serializeSlice(_reconstructed)));
      buf->PrependChain(std::move(serializeSlice(_volumeWeights)));

      _phase_performance[GAUSSIAN_RECONSTRUCTION].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);

}
/* End of GaussianReconstruction functions */

/*
 * SimulateSlices functions
 */
void irtkReconstruction::ParallelSimulateSlices() {

  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {

    auto workerId = _workers.at(workerIndex);

    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex, workerId]() {

      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      for (int inputIndex = start; inputIndex < end; ++inputIndex) {
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
      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

int irtkReconstruction::SimulateSlices(ebbrt::IOBuf::DataPointer& dp) {

  int initialize = dp.Get<int>();

  if (initialize) {
    _simulatedSlices.clear();
    _simulatedSlices.resize(_slices.size());

    _simulatedWeights.clear();
    _simulatedWeights.resize(_slices.size());

    _simulatedInside.clear();
    _simulatedInside.resize(_slices.size());

    for(int i= _start ; i < _end; i++) {
      _simulatedSlices[i] = _slices[i];
      _simulatedWeights[i] = _slices[i];
      _simulatedInside[i] = _slices[i];
    }
  }

  int reconSize = dp.Get<int>();
  dp.Get(reconSize*sizeof(double), (uint8_t*)_reconstructed.GetMat());

  ParallelSimulateSlices();

  return initialize;
}
/* End of SimulateSlices functions */

/*
 * InitializeRobustStatistics functions
 */

void irtkReconstruction::InitializeRobustStatistics(double& sigma, int& num) {
  int i, j;
  irtkRealImage slice, sim;
  sigma = 0.0;
  num = 0;

  for (unsigned int inputIndex = _start; inputIndex < _end; inputIndex++) {
    slice = _slices[inputIndex];

    // [fetalRecontruction] Voxel-wise sigma will be set to stdev of volumetric 
    // [fetalRecontruction] errors
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] calculate stev of the errors
          if ((_simulatedInside[inputIndex](i, j, 0) == 1) &&
              (_simulatedWeights[inputIndex](i, j, 0) > 0.99)) {
            slice(i, j, 0) -= _simulatedSlices[inputIndex](i, j, 0);
            sigma += slice(i, j, 0) * slice(i, j, 0);
            num++;
          }
        }

    // [fetalRecontruction] if slice does not have an overlap with ROI, 
    // [fetalRecontruction] set its weight to zero
    if (!_sliceInsideCPU[inputIndex])
      _sliceWeightCPU[inputIndex] = 0;
  }

}

void irtkReconstruction::ReturnFromInitializeRobustStatistics(double& sigma, 
    int& num, Messenger::NetworkId frontEndNid) {
  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, sigma, num]() {
      auto buf = MakeUniqueIOBuf((2 * sizeof(int)) + (1 * sizeof(double)));
      auto dp = buf->GetMutDataPointer();

      dp.Get<int>() = INITIALIZE_ROBUST_STATISTICS;
      dp.Get<int>() = num;
      dp.Get<double>() = sigma;

      _phase_performance[INITIALIZE_ROBUST_STATISTICS].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}
/* End of RobustStatistics functions */

/*
 * EStep functions
 */
double irtkReconstruction::M(double m) {
  return m*_step;
}

double irtkReconstruction::G(double x, double s) {
  return _step*exp(-x*x / (2 * s)) / (sqrt(6.28*s));
}

void irtkReconstruction::ParallelEStep(
    struct eStepReturnParameters& parameters) {

  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {

    auto workerId = _workers.at(workerIndex);

    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex, &parameters]() {

      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      double sum = 0;
      double den = 0;
      double sum2 = 0;
      double den2 = 0;
      double maxs = 0;
      double mins = 1;

      for (int inputIndex = start; inputIndex < end; inputIndex++) {
        irtkRealImage slice = _slices[inputIndex];
        _weights[inputIndex] = 0;
        irtkRealImage &b = _bias[inputIndex];
        double scale = _scaleCPU[inputIndex];

        double num = 0;
        // [fetalRecontruction] Calculate error, voxel weights, and slice potential
        for (int i = 0; i < slice.GetX(); i++) {
          for (int j = 0; j < slice.GetY(); j++) {
            if (slice(i, j, 0) != -1) {
              // [fetalRecontruction] bias correct and scale the slice
              slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

              // [fetalRecontruction] number of volumetric voxels to which
              // [fetalRecontruction] current slice voxel contributes
              int n = _volcoeffs[inputIndex][i][j].size();

              // [fetalRecontruction] if n == 0, slice voxel has no overlap with 
              // [fetalRecontruction] volumetric ROI, do not process it

              if ((n > 0) &&
                  (_simulatedWeights[inputIndex](i, j, 0) > 0)) {
                slice(i, j, 0) -=
                  _simulatedSlices[inputIndex](i, j, 0);

                // [fetalRecontruction] calculate norm and voxel-wise weights
                // [fetalRecontruction] Gaussian distribution for inliers
                // (likelihood)
                double g = G(slice(i, j, 0), _sigmaCPU);
                // [fetalRecontruction] Uniform distribution for outliers
                // (likelihood)
                double m = M(_mCPU);

                // [fetalRecontruction] voxel_wise posterior
                double weight = g * _mixCPU / (g * _mixCPU + m * (1 - _mixCPU));
                _weights[inputIndex](i, j, 0) = weight;
                // [fetalRecontruction] calculate slice potentials
                if (_simulatedWeights[inputIndex](i, j, 0) > 0.99) {
                  _slicePotential[inputIndex] += (1.0 - weight) * (1.0 - weight);
                  num++;
                }
              } else {
                _weights[inputIndex](i, j, 0) = 0;
              }
            }
          }
        }

        // [fetalRecontruction] evaluate slice potential
        if (num > 0) {
          _slicePotential[inputIndex] = sqrt(_slicePotential[inputIndex] / num);
        } else {
          // [fetalRecontruction] slice has no unpadded voxels
          _slicePotential[inputIndex] = -1; 
        }

        //TODO: Force excluded has to be received in the CoeffInit Step
        //To force-exclude slices predefined by a user, set their potentials to -1
        //for (unsigned int i = 0; i < _force_excluded.size(); i++)
        //  _slicePotential[_force_excluded[i]] = -1;

        for(int i = 0; i < _smallSlices.size(); i++) {
          _slicePotential[_smallSlices[i]] = -1;
        }

        if ((_scaleCPU[inputIndex] < 0.2) || (_scaleCPU[inputIndex] > 5)) {
          _slicePotential[inputIndex] = -1;
        }

        if (_slicePotential[inputIndex] >= 0) {
          // calculate means
          sum += _slicePotential[inputIndex] * _sliceWeightCPU[inputIndex];
          den += _sliceWeightCPU[inputIndex];
          sum2 += _slicePotential[inputIndex] * 
            (1 - _sliceWeightCPU[inputIndex]);
          den2 += (1 - _sliceWeightCPU[inputIndex]);

          // calculate min and max of potentials in case means need to be initalized
          if (_slicePotential[inputIndex] > maxs)
            maxs = _slicePotential[inputIndex];
          if (_slicePotential[inputIndex] < mins)
            mins = _slicePotential[inputIndex];
        } 
      }

      {
        std::lock_guard<ebbrt::SpinLock> l(spinLock);
        parameters.sum += sum;
        parameters.den += den;
        parameters.sum2 += sum2;
        parameters.den2 += den2;
        if (mins < parameters.mins)
          parameters.mins = mins;
        if (maxs > parameters.maxs)
          parameters.maxs = maxs;
      
      }
      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::StoreEStepParameters(
    ebbrt::IOBuf::DataPointer& dp) {
  auto parameters = dp.Get<struct eStepParameters>();
  _mCPU = parameters.mCPU;
  _sigmaCPU = parameters.sigmaCPU;
  _mixCPU = parameters.mixCPU;

  int smallSlicesSize = dp.Get<int>();
  _smallSlices.resize(smallSlicesSize);
  dp.Get(smallSlicesSize*sizeof(int), (uint8_t*) _smallSlices.data());
}

struct eStepReturnParameters irtkReconstruction::EStepI(
    ebbrt::IOBuf::DataPointer& dp) {
  StoreEStepParameters(dp);
  struct eStepReturnParameters parameters;
  parameters.sum = 0;
  parameters.den = 0;
  parameters.sum2 = 0;
  parameters.den2 = 0;
  parameters.maxs = 0;
  parameters.mins = 1;

  for (int i = 0; i < _slicePotential.size(); i++)
    _slicePotential[i] = 0;
  
  ParallelEStep(parameters);
  return parameters;
}

struct eStepReturnParameters irtkReconstruction::EStepII(
    ebbrt::IOBuf::DataPointer& dp) {

  auto parameters = dp.Get<struct eStepParameters>();
  double meanSCPU = parameters.meanSCPU;
  double meanS2CPU = parameters.meanS2CPU;
  
  struct eStepReturnParameters returnParameters;
  returnParameters.sum = 0;
  returnParameters.den = 0;
  returnParameters.sum2 = 0;
  returnParameters.den2 = 0;

  for (int inputIndex = _start; inputIndex < _end; inputIndex++) {
    if (_slicePotential[inputIndex] >= 0) {
      returnParameters.sum += (_slicePotential[inputIndex] - meanSCPU) *
        (_slicePotential[inputIndex] - meanSCPU) *
        _sliceWeightCPU[inputIndex];

      returnParameters.den += _sliceWeightCPU[inputIndex];

      returnParameters.sum2 += (_slicePotential[inputIndex] - meanS2CPU) *
        (_slicePotential[inputIndex] - meanS2CPU) *
        (1 - _sliceWeightCPU[inputIndex]);

      returnParameters.den2 += (1 - _sliceWeightCPU[inputIndex]);
    }
  }

  return returnParameters;
}

struct eStepReturnParameters irtkReconstruction::EStepIII(
    ebbrt::IOBuf::DataPointer& dp) {
  auto parameters = dp.Get<struct eStepParameters>();
  double sigmaSCPU = parameters.sigmaSCPU;
  double sigmaS2CPU = parameters.sigmaS2CPU;
  double meanSCPU = parameters.meanSCPU;
  double meanS2CPU = parameters.meanS2CPU;
  double den = parameters.den;
  double mixSCPU = parameters.mixSCPU;

  if (_debug) {
    cout << "[EStepIII input] _meanSCPU: " << meanSCPU << endl; 
    cout << "[EStepIII input] _meanS2CPU: " << meanS2CPU << endl; 
    cout << "[EStepIII input] _mixSCPU: " << mixSCPU << endl; 
    cout << "[EStepIII input] _sigmaSCPU: " << sigmaSCPU << endl; 
    cout << "[EStepIII input] _sigmaS2CPU: " << sigmaS2CPU << endl; 
    cout << "[EStepIII input] _den: " << den << endl; 
    PrintImageSums("[EStepIII input]");
  }

  struct eStepReturnParameters returnParameters;
  returnParameters.sum = 0;
  returnParameters.num = 0;

  double gs1, gs2;

  for (int inputIndex = _start; inputIndex < _end; inputIndex++) {
    // [fetalReconstruction] Slice does not have any voxels in volumetric ROI
    if (_slicePotential[inputIndex] == -1) {
      _sliceWeightCPU[inputIndex] = 0;
      continue;
    }

    // [fetalReconstruction] All slices are outliers or the means are not valid
    if ((den <= 0) || (meanS2CPU <= meanSCPU)) {
      _sliceWeightCPU[inputIndex] = 1;
      continue;
    }

    // [fetalReconstruction] likelihood for inliers
    if (_slicePotential[inputIndex] < meanS2CPU)
      gs1 = G(_slicePotential[inputIndex] - meanSCPU, sigmaSCPU);
    else
      gs1 = 0;

    // [fetalReconstruction] likelihood for outliers
    if (_slicePotential[inputIndex] > meanSCPU)
      gs2 = G(_slicePotential[inputIndex] - meanS2CPU, sigmaS2CPU);
    else
      gs2 = 0;

    // [fetalReconstruction] calculate slice weight
    double likelihood = gs1 * mixSCPU + gs2 * (1 - mixSCPU);
    if (likelihood > 0)
      _sliceWeightCPU[inputIndex] = gs1 * mixSCPU / likelihood;
    else {
      if (_slicePotential[inputIndex] <= meanSCPU)
        _sliceWeightCPU[inputIndex] = 1;
      if (_slicePotential[inputIndex] >= meanS2CPU)
        _sliceWeightCPU[inputIndex] = 0;
      if ((_slicePotential[inputIndex] < meanS2CPU) &&
          (_slicePotential[inputIndex] > meanSCPU)) // should not happen
        _sliceWeightCPU[inputIndex] = 1;
    }

    if (_slicePotential[inputIndex] >= 0) {
      returnParameters.sum += _sliceWeightCPU[inputIndex];
      returnParameters.num ++;
    }
  }
  return returnParameters;
}

void irtkReconstruction::ReturnFromEStepI(
    struct eStepReturnParameters parameters, Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, parameters]() {
      auto buf = MakeUniqueIOBuf(sizeof(int) + sizeof(eStepReturnParameters));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = E_STEP_I;
      dp.Get<struct eStepReturnParameters>() = parameters;

      _phase_performance[E_STEP_I].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}

void irtkReconstruction::ReturnFromEStepII(
    struct eStepReturnParameters parameters, Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, parameters]() {
      auto buf = MakeUniqueIOBuf(sizeof(int) + sizeof(eStepReturnParameters));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = E_STEP_II;
      dp.Get<struct eStepReturnParameters>() = parameters;

      _phase_performance[E_STEP_II].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}

void irtkReconstruction::ReturnFromEStepIII(
    struct eStepReturnParameters parameters, Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, parameters]() {
      auto buf = MakeUniqueIOBuf(sizeof(int) + sizeof(eStepReturnParameters));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = E_STEP_III;
      dp.Get<struct eStepReturnParameters>() = parameters;

      _phase_performance[E_STEP_III].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}
/* End of EStep functions */

/* 
 * Scale functions 
 */
void irtkReconstruction::ParallelScale() {

  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {

    auto workerId = _workers.at(workerIndex);
    
    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex]() {

      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      for (int inputIndex = start; inputIndex < end; inputIndex++) {
        // [fetalRecontruction] alias the current slice
        irtkRealImage &slice = _slices[inputIndex];

        // [fetalRecontruction] alias the current weight image
        irtkRealImage &w = _weights[inputIndex];

        // [fetalRecontruction] alias the current bias image
        irtkRealImage &b = _bias[inputIndex];

        // [fetalRecontruction] initialise calculation of scale
        double scalenum = 0;
        double scaleden = 0;

        for (int i = 0; i < slice.GetX(); i++)
          for (int j = 0; j < slice.GetY(); j++)
            if (slice(i, j, 0) != -1) {
              if (_simulatedWeights[inputIndex](i, j, 0) > 0.99) {
                // [fetalRecontruction] scale - intensity matching
                double eb = exp(-b(i, j, 0));
                scalenum += w(i, j, 0) * slice(i, j, 0) * eb *
                  _simulatedSlices[inputIndex](i, j, 0);
                scaleden += w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
              }
            }

        // [fetalRecontruction] calculate scale for this slice
        if (scaleden > 0)
          _scaleCPU[inputIndex] = scalenum / scaleden;
        else
          _scaleCPU[inputIndex] = 1;
      }
      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::Scale() {
  ParallelScale();
}

/* End of Scale functions */

/*
 * Superresolution functions
 */

void irtkReconstruction::ParallelSuperresolution() {
  
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {

    auto workerId = _workers.at(workerIndex);
    
    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex]() {
      
      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      irtkRealImage addon;
      irtkRealImage confidenceMap;

      addon.Initialize(_reconstructed.GetImageAttributes());
      confidenceMap.Initialize(_reconstructed.GetImageAttributes());

      addon = 0;
      confidenceMap = 0;

      for (int inputIndex = start; inputIndex < end; ++inputIndex) {
        // [fetalReconstruction] read the current slice
        irtkRealImage slice = _slices[inputIndex];

        // [fetalReconstruction] read the current weight image
        irtkRealImage &w = _weights[inputIndex];

        // [fetalReconstruction] read the current bias image
        irtkRealImage &b = _bias[inputIndex];

        // [fetalReconstruction] identify scale factor
        double scale = _scaleCPU[inputIndex];

        // [fetalReconstruction] Update reconstructed volume using current slice
        // [fetalReconstruction] Distribute error to the volume
        POINT3D p;
        for (int i = 0; i < slice.GetX(); i++) {
          for (int j = 0; j < slice.GetY(); j++) {
            if (slice(i, j, 0) != -1) {
              // [fetalReconstruction] bias correct and scale the slice
              slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

              if (_simulatedSlices[inputIndex](i, j, 0) > 0)
                slice(i, j, 0) -= _simulatedSlices[inputIndex](i, j, 0);
              else
                slice(i, j, 0) = 0;

              int n = _volcoeffs[inputIndex][i][j].size();
              for (int k = 0; k < n; k++) {
                p = _volcoeffs[inputIndex][i][j][k];
                addon(p.x, p.y, p.z) += p.value * slice(i, j, 0) * w(i, j, 0) *
                  _sliceWeightCPU[inputIndex];
                confidenceMap(p.x, p.y, p.z) += p.value * w(i, j, 0) *
                  _sliceWeightCPU[inputIndex];
              }
            }
          }
        }
      }
      
      {
        std::lock_guard<ebbrt::SpinLock> l(spinLock);
        _addon += addon;
        _confidenceMap += confidenceMap;
      }

      count++;
      bar.Wait();
      while(count < _workers.size()); 

      
    if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));

    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::SuperResolution(ebbrt::IOBuf::DataPointer& dp) {

  int iter = dp.Get<int>();
  
  if(iter == 1) {
      _addon.Initialize(_reconstructed.GetImageAttributes());
      _confidenceMap.Initialize(_reconstructed.GetImageAttributes());
  } 
  // Clear addon
  _addon = 0;

  // Clear confidence map
  _confidenceMap = 0;
  
  ParallelSuperresolution();

}

void irtkReconstruction::ReturnFromSuperResolution(
    Messenger::NetworkId frontEndNid) {
  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid]() {
      auto buf = MakeUniqueIOBuf(sizeof(int));
      auto dp = buf->GetMutDataPointer();
  
      dp.Get<int>() = SUPERRESOLUTION;
  
      buf->PrependChain(std::move(serializeSlice(_addon)));
      buf->PrependChain(std::move(serializeSlice(_confidenceMap)));

      _phase_performance[SUPERRESOLUTION].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}

/* End of Superresolution */

/*
 * MStep functions
 */
void irtkReconstruction::ParallelMStep(mStepReturnParameters& parameters) {
  
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());


  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {
    
    auto workerId = _workers.at(workerIndex);
    
    ebbrt::event_manager->SpawnRemote(
      [this, &context, &count, mainCPU, workerIndex, workerId, &parameters]() {
      
      int start = workerIndex * _factor + _start;
      int end = start + _factor; 
      end = end > _end ? _end : end;

      double sigma = 0;
      double mix = 0;
      double min = 0;
      double max = 0;
      int num = 0;
    
      for (int inputIndex = start; inputIndex < end; ++inputIndex) {

        irtkRealImage slice = _slices[inputIndex];

        irtkRealImage &w = _weights[inputIndex];

        irtkRealImage &b = _bias[inputIndex];

        // [fetalReconstruction] identify scale factor
        double scale = _scaleCPU[inputIndex];

        // [fetalReconstruction] calculate error
        for (int i = 0; i < slice.GetX(); i++) {
          for (int j = 0; j < slice.GetY(); j++) {
            if (slice(i, j, 0) != -1) {
              // [fetalReconstruction] bias correct and scale the slice
              slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

              // [fetalReconstruction] otherwise the error has no meaning - 
              // [fetalReconstruction] it is equal to slice intensity
              if (_simulatedWeights[inputIndex](i, j, 0) > 0.99) {

                slice(i, j, 0) -= _simulatedSlices[inputIndex](i, j, 0);

                double e = slice(i, j, 0);
                sigma += e * e * w(i, j, 0);
                mix += w(i, j, 0);

                if (e < min)
                  min = e;
                if (e > max)
                  max = e;

                num++;
              }
            }
          }
        }
      } 

      {
        
        std::lock_guard<ebbrt::SpinLock> l(spinLock);
        parameters.sigma += sigma;
        parameters.mix += mix;
        parameters.num += num;
        if (min < parameters.min)
          parameters.min = min;
        if (max > parameters.max)
          parameters.max = max;
      }

      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
      }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::MStep(mStepReturnParameters& parameters) {
  parameters.sigma = 0;
  parameters.mix = 0;
  parameters.num = 0;
  parameters.min = voxel_limits<irtkRealPixel>::max();
  parameters.max = voxel_limits<irtkRealPixel>::min();

  ParallelMStep(parameters);
}

void irtkReconstruction::ReturnFromMStep(mStepReturnParameters& parameters,
    Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, parameters]() {
      auto buf = MakeUniqueIOBuf(sizeof(int) + sizeof(mStepReturnParameters));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = M_STEP;
      dp.Get<mStepReturnParameters>() = parameters;
      _phase_performance[M_STEP].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}
/* End of MStep*/

/*
 * RestoreSliceIntensities functions
 */

void irtkReconstruction::RestoreSliceIntensities() {
  double factor;
  irtkRealPixel *p;
  for (int inputIndex = _start; inputIndex < _end; inputIndex++) {
      // [fetalRecontruction] calculate scaling factor 
      // [fetalRecontruction] _average_value;
      factor = _stackFactor[_stackIndex[inputIndex]];
      // [fetalRecontruction] read the pointer to current slice
      p = _slices[inputIndex].GetPointerToVoxels();
      for (int i = 0; i < _slices[inputIndex].GetNumberOfVoxels(); i++) {
        if (*p > 0)
          *p = *p / factor;
        p++;
      }
  }
}

/* End of RestoreSliceIntensities*/

/*
 * ScaleVolume functions
 */

struct scaleVolumeParameters irtkReconstruction::ScaleVolume() {
  scaleVolumeParameters parameters;
  parameters.num = 0;
  parameters.den = 0;

  for (int inputIndex = _start; inputIndex < _end; inputIndex++) {
    irtkRealImage &slice = _slices[inputIndex];
    irtkRealImage &w = _weights[inputIndex];
    irtkRealImage &sim = _simulatedSlices[inputIndex];

    for (int i = 0; i < slice.GetX(); i++) {
      for (int j = 0; j < slice.GetY(); j++) {
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] scale - intensity matching
          if (_simulatedWeights[inputIndex](i, j, 0) > 0.99) {
            parameters.num += w(i, j, 0) * _sliceWeightCPU[inputIndex] *
              slice(i, j, 0) * sim(i, j, 0);
            parameters.den += w(i, j, 0) * _sliceWeightCPU[inputIndex] *
              sim(i, j, 0) * sim(i, j, 0);
          }
        }
      }
    }
  } 
  return parameters;
}

void irtkReconstruction::ReturnFromScaleVolume(
    struct scaleVolumeParameters parameters, Messenger::NetworkId frontEndNid) {
  
  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid, parameters]() {
      auto buf = MakeUniqueIOBuf(sizeof(int) + sizeof(scaleVolumeParameters));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = SCALE_VOLUME;
      dp.Get<scaleVolumeParameters>() = parameters;
      _phase_performance[SCALE_VOLUME].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);

}
/* End of ScaleVolume */

/*
 * SliceToVolumeRegistration functions
 */

void irtkReconstruction::ParallelSliceToVolumeRegistration() {
  irtkImageAttributes attr = _reconstructed.GetImageAttributes();
  
  size_t mainCPU = ebbrt::Cpu::GetMine();
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  static ebbrt::SpinBarrier bar(_workers.size());

  for (size_t workerIndex = 0; workerIndex < _workers.size(); workerIndex++) {
    
      auto workerId = _workers.at(workerIndex);
    
      ebbrt::event_manager->SpawnRemote(
              [this, &context, &count, mainCPU, workerIndex, attr]() {
        int start = workerIndex * _factor + _start;
        int end = start + _factor; 
        end = end > _end ? _end : end;

        for (int inputIndex = start; inputIndex < end; inputIndex++) {
            irtkImageRigidRegistrationWithPadding registration;
            irtkGreyPixel smin, smax;
            irtkGreyImage target;
            irtkRealImage slice, w, b, t;
            irtkResamplingWithPadding<irtkRealPixel> resampling(attr._dx, attr._dx,
                                                                attr._dx, -1);

            t = _slices[inputIndex];
            resampling.SetInput(&_slices[inputIndex]);
            resampling.SetOutput(&t);
            resampling.Run();
            target = t;
        target.GetMinMax(&smin, &smax);

        if (smax > -1) {
          // [fetalRecontruction] put origin to zero
          irtkRigidTransformation offset;
          ResetOrigin(target, offset);
          irtkMatrix mo = offset.GetMatrix();
          irtkMatrix m = _transformations[inputIndex].GetMatrix();
          m = m * mo;
          _transformations[inputIndex].PutMatrix(m);

          irtkGreyImage source = _reconstructed;
          registration.SetInput(&target, &source);
          
          registration.SetOutput(&_transformations[inputIndex]);
          registration.GuessParameterSliceToVolume();
          registration.SetTargetPadding(-1);
          
          /*
             if (_debug) {
               cout << "[ParallelSliceToVolumeRegistration input] " << inputIndex
               << " transformation: ";
               _transformations[inputIndex].Print2();
               cout << endl;
               }
          */

          registration.Run();
          
          /*
             if (_debug) {
               cout << "[ParallelSliceToVolumeRegistration output] " << inputIndex 
               << " transformation: ";
               _transformations[inputIndex].Print2();
               cout << endl;
               }
          */
          
          // [fetalRecontruction] undo the offset
          mo.Invert();
          m = _transformations[inputIndex].GetMatrix();
          m = m * mo;
          _transformations[inputIndex].PutMatrix(m);
          
        }
      }
      count++;
      bar.Wait();
      while(count < _workers.size()); 
      if (ebbrt::Cpu::GetMine() == mainCPU)
        ebbrt::event_manager->ActivateContext(std::move(context));
    }, workerId);
  }
  ebbrt::event_manager->SaveContext(context);
}

void irtkReconstruction::SliceToVolumeRegistration(
    ebbrt::IOBuf::DataPointer& dp) {
  
  int reconSize = dp.Get<int>();
  dp.Get(reconSize*sizeof(double), (uint8_t*)_reconstructed.GetMat());

  ParallelSliceToVolumeRegistration();
}

void irtkReconstruction::ReturnFromSliceToVolumeRegistration(
    Messenger::NetworkId frontEndNid) {
  
  ebbrt::event_manager->SpawnRemote(
      [this,frontEndNid]() {
      auto buf = MakeUniqueIOBuf(3*sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = SLICE_TO_VOLUME_REGISTRATION;
      dp.Get<int>() = _start;
      dp.Get<int>() = _end;
	
      //TODO: Serialize only the transformations for that backend and not
      //all of them
      buf->PrependChain(std::move(serializeTransformations(_transformations)));

      _phase_performance[SLICE_TO_VOLUME_REGISTRATION].sent += buf->ComputeChainDataLength();
      SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}
/* End of SliceToVolumeRegistration */

void irtkReconstruction::ReturnFrom(int fn, Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
        [this,frontEndNid, fn]() {
        auto buf = MakeUniqueIOBuf(sizeof(int));
        auto dp = buf->GetMutDataPointer();
        dp.Get<int>() = fn;
        _phase_performance[fn].sent += sizeof(int); 
        SendMessage(frontEndNid, std::move(buf));
  }, _IOCPU);
}

void irtkReconstruction::SendTimers(Messenger::NetworkId frontEndNid) {

  ebbrt::event_manager->SpawnRemote(
      [this, frontEndNid]() {
        auto buf =
            MakeUniqueIOBuf(sizeof(int) + sizeof(phases_data));
        auto dp = buf->GetMutDataPointer();
        dp.Get<int>() = GATHER_TIMERS;
        dp.Get<phases_data>() = _phase_performance;
        SendMessage(frontEndNid, std::move(buf));
      },
      _IOCPU);
}

void irtkReconstruction::ExecuteCoeffInit(ebbrt::IOBuf::DataPointer& dp, 
    size_t cpu) {

  auto start = startTimer();
  CoeffInit(dp, cpu);
  auto seconds = endTimer(start);
  _phase_performance[COEFF_INIT].time += seconds; 

  if (_debug) {
    cout << "[CoeffInit output] _averageVolumeWeight: " 
      << _averageVolumeWeight << endl;
    PrintImageSums("[CoeffInit output]");
    cout << "[CoeffInit time] " << seconds << endl;
  }
}

void irtkReconstruction::ExecuteGaussianReconstruction(
    Messenger::NetworkId frontEndNid) {


  cout << "In ExecuteGaussianReconstruction() with frontEnd network " << frontEndNid.ToString() << endl;

  auto start = startTimer();
  GaussianReconstruction();
  ReturnFromGaussianReconstruction(frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[GAUSSIAN_RECONSTRUCTION].time += seconds; 

  if (_debug) {
    PrintImageSums("[GaussianReconstruction output]");
    cout << fixed << "[GaussianReconstruction output] _volumeWeights: " 
      << SumImage(_volumeWeights) << endl;
    cout << "[GaussianReconstruction time] " << seconds << endl; 
  }
}

int irtkReconstruction::ExecuteSimulateSlices(ebbrt::IOBuf::DataPointer& dp) { 

  auto start = startTimer();
  auto initialize = SimulateSlices(dp);
  auto seconds = endTimer(start);
  _phase_performance[SIMULATE_SLICES].time += seconds; 

  if (_debug) {
    PrintImageSums("[SimulateSlices output]");
    cout << "[SimulateSlices output] " << seconds << endl;
  }

  return initialize;
}

void irtkReconstruction::ExecuteInitializeRobustStatistics(Messenger::NetworkId frontEndNid) { 
  auto start = startTimer();
  double sigma;
  int num;
  InitializeRobustStatistics(sigma, num);
  ReturnFromInitializeRobustStatistics(sigma, num, frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[INITIALIZE_ROBUST_STATISTICS].time += seconds; 

  if (_debug) {
    cout << "[InitializeRobustStatistics output] sigma: " << sigma << endl;
    cout << "[InitializeRobustStatistics output] num: " << num << endl; 
    cout << "[InitializeRobustStatistics time]  " << seconds << endl; 
  }
}

void irtkReconstruction::ExecuteMStep(Messenger::NetworkId frontEndNid) {
  auto start = startTimer();
  mStepReturnParameters parameters;
  MStep(parameters);
  ReturnFromMStep(parameters, frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[M_STEP].time += seconds; 
          
  if (_debug) {
    cout << "[MStep output] sigma: " << parameters.sigma << endl;
    cout << "[MStep output] mix: " << parameters.mix << endl;
    cout << "[MStep output] num: " << parameters.num << endl;
    cout << "[MStep output] min: " << parameters.min << endl;
    cout << "[MStep output] max: " << parameters.max << endl;
    cout << "[MStep time] " << seconds << endl;
  }
}

void irtkReconstruction::ExecuteEStepI(ebbrt::IOBuf::DataPointer& dp, 
    Messenger::NetworkId frontEndNid) { 

  auto start = startTimer();
  auto parameters = EStepI(dp);
  ReturnFromEStepI(parameters, frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[E_STEP_I].time += seconds; 

  if (_debug)
    cout << "[EStepI time] " << seconds << endl;
}

void irtkReconstruction::ExecuteEStepII(ebbrt::IOBuf::DataPointer& dp, 
    Messenger::NetworkId frontEndNid) { 
  auto start = startTimer();

  auto parameters = EStepII(dp);
  ReturnFromEStepII(parameters, frontEndNid);

  auto seconds = endTimer(start);
  _phase_performance[E_STEP_II].time += seconds; 

  if (_debug)
    cout << "[EStepII time] " << seconds << endl;
}

void irtkReconstruction::ExecuteEStepIII(ebbrt::IOBuf::DataPointer& dp, 
    Messenger::NetworkId frontEndNid) { 

  auto start = startTimer();
  auto parameters = EStepIII(dp);
  ReturnFromEStepIII(parameters, frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[E_STEP_III].time += seconds; 

  if (_debug)
    cout << "[EStepIII time] " << seconds << endl;
}

void irtkReconstruction::ExecuteScale(Messenger::NetworkId frontEndNid) { 

  auto start = startTimer();
  Scale();
  ReturnFrom(SCALE, frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[SCALE].time += seconds; 

  if (_debug) {
    PrintImageSums("[Scale output]");
    cout << "[Scale time] " << seconds << endl;
  }
}

void irtkReconstruction::ExecuteSuperResolution(ebbrt::IOBuf::DataPointer& dp, 
    Messenger::NetworkId frontEndNid) { 

  auto start = startTimer();
  SuperResolution(dp);
  ReturnFromSuperResolution(frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[SUPERRESOLUTION].time += seconds; 

  if (_debug) {
    cout << fixed << "[SuperResolution output] _addon: " 
      << SumImage(_addon) << endl;
    cout << fixed << "[SuperResolution output] _confidenceMap: " 
      << SumImage(_confidenceMap) << endl;
    cout << "[SuperResolution time] " << seconds << endl;
  }
}

void irtkReconstruction::ExecuteRestoreSliceIntensities() {

  auto start = startTimer();
  RestoreSliceIntensities();
  auto seconds = endTimer(start);
  _phase_performance[RESTORE_SLICE_INTENSITIES].time += seconds; 

  if (_debug)
    cout << "[RestoreSliceIntensities time] " << seconds << endl;
}

void irtkReconstruction::ExecuteScaleVolume(Messenger::NetworkId nid) {

  auto start = startTimer();
  auto parameters = ScaleVolume();
  ReturnFromScaleVolume(parameters, nid);
  auto seconds = endTimer(start);
  _phase_performance[SCALE_VOLUME].time += seconds; 

  if (_debug)
    cout << "[ScaleVolume time] " << seconds << endl;
}

void irtkReconstruction::ExecuteSliceToVolumeRegistration(
    ebbrt::IOBuf::DataPointer& dp, Messenger::NetworkId frontEndNid) { 

  auto start = startTimer();
  SliceToVolumeRegistration(dp);
  ReturnFromSliceToVolumeRegistration(frontEndNid);
  auto seconds = endTimer(start);
  _phase_performance[SLICE_TO_VOLUME_REGISTRATION].time += seconds; 

  if (_debug)
    cout << "[SliceToVolumeRegistration time] " << seconds << endl;
}

void irtkReconstruction::ReceiveMessage (Messenger::NetworkId nid,
    std::unique_ptr<IOBuf> &&buffer) {
  size_t cpu = ebbrt::Cpu::GetMine();

  auto targetCpu = (cpu + 1) % ebbrt::Cpu::Count();

  string nidStr = nid.ToString();
  cout << "Receiving message on network: " << nid.ToString() << " data of size: " << buffer->ComputeChainDataLength() << endl;

  ebbrt::event_manager->SpawnRemote(
    [this, buffer = std::move(buffer), nid, nidStr, cpu]() {

    auto len = buffer->ComputeChainDataLength();
    auto dp = buffer->GetDataPointer();
    auto fn = dp.Get<int>();
    if (fn < WORK_PHASES)
      _phase_performance[fn].recv += len; 

    if (_debug) {
      cout << "Receiving function: " << fn << " on CPU: " 
        << ebbrt::Cpu::GetMine() <<  endl;
    }
    switch(fn) {
      case COEFF_INIT:
        {
          ExecuteCoeffInit(dp, cpu);
          ExecuteGaussianReconstruction(nid);
          break;
        }
      case SIMULATE_SLICES:
        {
          auto initialize = ExecuteSimulateSlices(dp);
          if (initialize)
            ExecuteInitializeRobustStatistics(nid);
          else
            ExecuteMStep(nid);
          break;
        }
      case E_STEP_I:
        {
          ExecuteEStepI(dp, nid); 
          break;
        }
      case E_STEP_II:
        {
          ExecuteEStepII(dp, nid);
          break;
        }
      case E_STEP_III:
        {
          ExecuteEStepIII(dp, nid);
          break;
        }
      case SCALE:
        {
          ExecuteScale(nid);
          break;
        }
      case SUPERRESOLUTION:
        {
          ExecuteSuperResolution(dp, nid);
          break;
        }
      case RESTORE_SLICE_INTENSITIES:
        {
          ExecuteRestoreSliceIntensities(); 
          ExecuteScaleVolume(nid);
          break;
        }
      case SLICE_TO_VOLUME_REGISTRATION:
        {
          ExecuteSliceToVolumeRegistration(dp, nid); 
          break;
        }
      case GATHER_TIMERS:
        {
          SendTimers(nid);
          break;
        }
      case PING:
        {
          cout << "recevied ping message from " << nidStr << endl;
          break;
        }
      default:
        cout << "Invalid option" << endl;
    }
  }, targetCpu); // End of SpawnRemote
}

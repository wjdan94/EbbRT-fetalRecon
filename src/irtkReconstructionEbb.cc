/*=========================================================================
  Library   : Image Registration Toolkit (IRTK)
  Copyright : Imperial College, Department of Computing
  Visual Information Processing (VIP), 2011 onwards
  Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
  Version   : $Revision: 1 $
  Changes   : $Author: bkainz $

  Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
  Maria Murgasova, Kevin Keraudren
  All rights reserved.

  If you use this work for research we would very much appreciate if you cite
  Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina
  Malamateniou,
  Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
  Joseph V. Hajnal and Daniel Rueckert:
  Fast Volume Reconstruction from Motion Corrupted 2D Slices.
  IEEE Transactions on Medical Imaging, in press, 2015

  IRTK IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
  COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
  COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN
  AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

  BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
  TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
  CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
  HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
  FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  =========================================================================*/

#define NOMINMAX
#define _USE_MATH_DEFINES


#include "irtkReconstructionEbb.h"
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include <thread>

#include <ebbrt/EbbRef.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/Message.h>
#include <ebbrt/SharedEbb.h>
#include <ebbrt/SpinBarrier.h>
#include <ebbrt/StaticIOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/Future.h>

#ifdef __EBBRT_BM__
#include <ebbrt/SpinLock.h>
#include <ebbrt/native/Clock.h>
#endif

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, irtkReconstructionEbb);

using namespace ebbrt;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef __EBBRT_BM__
#define PRINTF ebbrt::kprintf
#define FORPRINTF ebbrt::force_kprintf
static size_t indexToCPU(size_t i) { return i; }
ebbrt::SpinLock spinlock;
#else
#define PRINTF std::printf
#define FORPRINTF std::printf
#endif

void irtkReconstructionEbb::printvolcoeffs() {
  int i, j, n, k, inputIndex;
  POINT3D p;
  double sum = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double max_x, max_y, max_z, min_x, min_y, min_z;

  max_x = max_y = max_z = DBL_MIN;
  min_x = min_y = min_z = DBL_MAX;

  for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
    for (i = 0; i < _slices[inputIndex].GetX(); i++) {
      for (j = 0; j < _slices[inputIndex].GetY(); j++) {
        n = _volcoeffs[inputIndex][i][j].size();

        for (k = 0; k < n; k++) {
          p = _volcoeffs[inputIndex][i][j][k];
          if (p.x > max_x)
            max_x = p.x;
          if (p.y > max_y)
            max_y = p.y;
          if (p.z > max_z)
            max_z = p.z;

          if (p.x < min_x)
            min_x = p.x;
          if (p.y < min_y)
            min_y = p.y;
          if (p.z < min_z)
            min_z = p.z;

          x += (double)p.x;
          y += (double)p.y;
          z += (double)p.z;
          sum += (double)p.value;
        }
      }
    }
  }

  //FORPRINTF("x:%lf y:%lf z:%lf value:%lf\n", x, y, z, sum);
  //FORPRINTF("max_x:%lf max_y:%lf max_z:%lf, min_x:%lf min_y:%lf min_z:%lf\n",
  //          max_x, max_y, max_z, min_x, min_y, min_z);
}

void printStackAttr(irtkRealImage s) {
  irtkImageAttributes at;

  at = s.GetImageAttributes();
  //FORPRINTF("%lf\n", at.Sum());

  /*FORPRINTF("%d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf
     %lf %lf %lf %lf\n",
            at._x,
            at._y,
            at._z,
            at._t,
            // Default voxel size
            at._dx,
            at._dy,
            at._dz,
            at._dt,

            // Default origin
            at._xorigin,
            at._yorigin,
            at._zorigin,
            at._torigin,

            // Default x-axis
            at._xaxis[0],
            at._xaxis[1],
            at._xaxis[2],

            // Default y-axis
            at._yaxis[0],
            at._yaxis[1],
            at._yaxis[2],

            // Default z-axis
            at._zaxis[0],
            at._zaxis[1],
            at._zaxis[2]);*/
}

void printI2W(irtkRealImage s) {
  auto mat = s.GetImageToWorldMatrix();
  //FORPRINTF("rows = %d cols = %d sum = %lf\n", mat.Rows(), mat.Cols(), mat.Sum());
}

void printW2I(irtkRealImage s) {
  auto mat = s.GetWorldToImageMatrix();
  //FORPRINTF("rows = %d cols = %d sum = %lf\n", mat.Rows(), mat.Cols(), mat.Sum());
}

void printSlices(irtkRealImage s) {
    /*double *ptr = s.GetMat();
    int n = s.GetSizeMat();
    double sum = 0.0;
    
    for (int j = 0; j < n; j++) {
	sum += ptr[j];
    }
    
    FORPRINTF("_matrix n = %d, sum = %lf\n", n, sum);*/
    //FORPRINTF("n = %d, sum = %lf\n", s.GetNumberOfVoxels(), s.Sum());
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageAttr(irtkRealImage ri) {
  irtkImageAttributes at;

  at = ri.GetImageAttributes();

  auto buf = MakeUniqueIOBuf((4 * sizeof(int)) + (17 * sizeof(double)));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = at._x;
  dp.Get<int>() = at._y;
  dp.Get<int>() = at._z;
  dp.Get<int>() = at._t;

  // Default voxel size
  dp.Get<double>() = at._dx;
  dp.Get<double>() = at._dy;
  dp.Get<double>() = at._dz;
  dp.Get<double>() = at._dt;

  // Default origin
  dp.Get<double>() = at._xorigin;
  dp.Get<double>() = at._yorigin;
  dp.Get<double>() = at._zorigin;
  dp.Get<double>() = at._torigin;

  // Default x-axis
  dp.Get<double>() = at._xaxis[0];
  dp.Get<double>() = at._xaxis[1];
  dp.Get<double>() = at._xaxis[2];

  // Default y-axis
  dp.Get<double>() = at._yaxis[0];
  dp.Get<double>() = at._yaxis[1];
  dp.Get<double>() = at._yaxis[2];

  // Default z-axis
  dp.Get<double>() = at._zaxis[0];
  dp.Get<double>() = at._zaxis[1];
  dp.Get<double>() = at._zaxis[2];

  return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageI2W(irtkRealImage& ri) {
  auto buf = MakeUniqueIOBuf(2 * sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Rows();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Cols();

  auto buf2 = std::make_unique<StaticIOBuf>(
      reinterpret_cast<const uint8_t *>(ri.GetWorldToImageMatrix().GetMatrix()),
      (size_t)(ri.GetWorldToImageMatrix().Rows() * ri.GetWorldToImageMatrix().Cols() * sizeof(double)));
  buf->PrependChain(std::move(buf2));

  return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageW2I(irtkRealImage& ri) {
    auto buf = MakeUniqueIOBuf(2 * sizeof(int));
    auto dp = buf->GetMutDataPointer();
    dp.Get<int>() = ri.GetWorldToImageMatrix().Rows();
    dp.Get<int>() = ri.GetWorldToImageMatrix().Cols();
    
    auto buf2 = std::make_unique<StaticIOBuf>(
	reinterpret_cast<const uint8_t *>(ri.GetWorldToImageMatrix().GetMatrix()),
	(size_t)(ri.GetWorldToImageMatrix().Rows() * ri.GetWorldToImageMatrix().Cols() * sizeof(double)));
    buf->PrependChain(std::move(buf2));
    
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeSlices(irtkRealImage& ri) {
    auto buf = MakeUniqueIOBuf(1 * sizeof(int));
    auto dp = buf->GetMutDataPointer();
    dp.Get<int>() = ri.GetSizeMat();
    
    auto buf2 = std::make_unique<StaticIOBuf>(
	reinterpret_cast<const uint8_t *>(ri.GetMat()),
	(size_t)(ri.GetSizeMat() * sizeof(double)));
    
    buf->PrependChain(std::move(buf2));
    
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeRigidTransMat(irtkRigidTransformation& rt) {
    auto buf = MakeUniqueIOBuf(2 * sizeof(int));
    auto dp = buf->GetMutDataPointer();
    dp.Get<int>() = rt._matrix.Rows();
    dp.Get<int>() = rt._matrix.Cols();

    auto buf2 = std::make_unique<StaticIOBuf>(
	reinterpret_cast<const uint8_t *>(rt._matrix.GetMatrix()),
	(size_t)(rt._matrix.Rows() * rt._matrix.Cols()  * sizeof(double)));
    
    buf->PrependChain(std::move(buf2));
    
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeRigidTrans(irtkRigidTransformation& rt) {
    auto buf = MakeUniqueIOBuf((12 * sizeof(double)) + (8 * sizeof(int)));
    auto dp = buf->GetMutDataPointer();
    
    dp.Get<double>() = rt._tx;
    dp.Get<double>() = rt._ty;
    dp.Get<double>() = rt._tz;
    
    dp.Get<double>() = rt._rx;
    dp.Get<double>() = rt._ry;
    dp.Get<double>() = rt._rz;
    
    dp.Get<double>() = rt._cosrx;
    dp.Get<double>() = rt._cosry;
    dp.Get<double>() = rt._cosrz;

    dp.Get<double>() = rt._sinrx;
    dp.Get<double>() = rt._sinry;
    dp.Get<double>() = rt._sinrz;

    dp.Get<int>() = (int)(rt._status[0]);
    dp.Get<int>() = (int)(rt._status[1]);
    dp.Get<int>() = (int)(rt._status[2]);
    dp.Get<int>() = (int)(rt._status[3]);
    dp.Get<int>() = (int)(rt._status[4]);
    dp.Get<int>() = (int)(rt._status[5]);
			  
    dp.Get<int>() = rt._matrix.Rows();
    dp.Get<int>() = rt._matrix.Cols();
    
    auto buf2 = std::make_unique<StaticIOBuf>(
	reinterpret_cast<const uint8_t *>(rt._matrix.GetMatrix()),
	(size_t)(rt._matrix.Rows() * rt._matrix.Cols()  * sizeof(double)));
    
    buf->PrependChain(std::move(buf2));
    
    return buf;
}

double sumTrans(std::vector<irtkRigidTransformation> _transformations, int s, int e)
{
    double sum = 0.0;
    for (int i = s; i < e; i++) {
	sum += _transformations[i].Sum();
    }
    return sum;
}

double sumTrans2(std::vector<irtkRigidTransformation> _transformations, int s, int e)
{
    double sum = 0.0;
    for (int i = s; i < e; i++) {
	sum += _transformations[i]._matrix.Sum();
    }
    return sum;
}


double sumOneImage(irtkRealImage a) {
  double sum = 0.0;
  irtkRealPixel *ap = a.GetPointerToVoxels();

  for (int j = 0; j < a.GetNumberOfVoxels(); j++) {
    sum += (double)*ap;
    ap++;
  }
  return sum;
}

double sumImage(std::vector<irtkRealImage> a) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); i++) {
      sum += a[i].Sum();
  }
  return sum;
}

double sumPartImage(std::vector<irtkRealImage> a, int s, int e) {
  double sum = 0.0;
  for (int i = s; i < e; i++) {
      sum += a[i].Sum();
  }
  return sum;
}

double sumVec(std::vector<double> b) {
  double sum = 0.0;
  for (int i = 0; i < b.size(); i++) {
    sum += (double)b[i];
  }
  return sum;
}

float sumFVec(std::vector<float> b) {
  float sum = 0.0;
  for (int i = 0; i < b.size(); i++) {
    sum += b[i];
  }
  return sum;
}

int sumBool(std::vector<bool> b) {
  int sum = 0;
  for (int i = 0; i < b.size(); i++) {
    if (b[i])
      sum++;
  }
  return sum;
}

int sumInt(std::vector<int> b) {
  int sum = 0;
  for (int i = 0; i < b.size(); i++) {
    if (b[i])
      sum++;
  }
  return sum;
}

int sumPartInt(std::vector<int> b, int s, int e) {
  int sum = 0;
  for (int i = s; i < e; i++) {
    if (b[i])
      sum++;
  }
  return sum;
}

double sumPartVec(std::vector<double> b, int s, int e) {
  double sum = 0.0;
  for (int i = s; i < e; i++) {
    sum += (double)b[i];
  }
  return sum;
}

struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

void irtkReconstructionEbb::Print(ebbrt::Messenger::NetworkId nid,
                                  const char *str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char *>(buf->MutData()), len, "%s", str);

#ifndef __EBBRT_BM__
  std::cout << "[H] EbbRTReconstruction length of sent iobuf: "
            << buf->ComputeChainDataLength() << " bytes" << std::endl;
#else
  ebbrt::kprintf("[B] EbbRTReconstruction length of sent iobuf: %ld bytes \n",
                 buf->ComputeChainDataLength());
#endif

  SendMessage(nid, std::move(buf));
}

void irtkReconstructionEbb::setNumNodes(int i) { numNodes = i; }
void irtkReconstructionEbb::setNumThreads(int i) { _numThreads = i; }

irtkReconstructionEbb::irtkReconstructionEbb(EbbId ebbid)
    : Messagable<irtkReconstructionEbb>(ebbid) {
  _step = 0.0001;
  _debug = false;
  _quality_factor = 2;
  _sigma_bias = 12;
  _sigma_s_cpu = 0.025;
  _sigma_s_gpu = 0.025;
  _sigma_s2_cpu = 0.025;
  _sigma_s2_gpu = 0.025;
  _mix_s_cpu = 0.9;
  _mix_s_gpu = 0.9;
  _mix_cpu = 0.9;
  _mix_gpu = 0.9;
  _delta = 1;
  _lambda = 0.1;
  _alpha = (0.05 / _lambda) * _delta * _delta;
  _template_created = false;
  _have_mask = false;
  _low_intensity_cutoff = 0.01;
  _global_bias_correction = false;
  _adaptive = false;
  _use_SINC = false;

  int directions[13][3] = {{1, 0, -1}, {0, 1, -1}, {1, 1, -1}, {1, -1, -1},
                           {1, 0, 0},  {0, 1, 0},  {1, 1, 0},  {1, -1, 0},
                           {1, 0, 1},  {0, 1, 1},  {1, 1, 1},  {1, -1, 1},
                           {0, 0, 1}};
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 3; j++)
      _directions[i][j] = directions[i][j];

  _useCPUReg = true;
  _useCPU = true;

  nids.clear();
  reconRecv = 0;

  _tsigma = 0;
  tmix = 0;
  _tnum = 0;
  tmin = voxel_limits<irtkRealPixel>::max();
  tmax = voxel_limits<irtkRealPixel>::min();

  bytesTotal = 0;
}

EbbRef<irtkReconstructionEbb> irtkReconstructionEbb::Create(EbbId id) {
  return EbbRef<irtkReconstructionEbb>(id);
}

// This Ebb is implemented with one representative per machine
irtkReconstructionEbb &irtkReconstructionEbb::HandleFault(EbbId id) {
  {
    // First we check if the representative is in the LocalIdMap (using a
    // read-lock)
    LocalIdMap::ConstAccessor accessor;
    auto found = local_id_map->Find(accessor, id);
    if (found) {
      auto &rep = *boost::any_cast<irtkReconstructionEbb *>(accessor->second);
      EbbRef<irtkReconstructionEbb>::CacheRef(id, rep);
      return rep;
    }
  }

  irtkReconstructionEbb *rep;
  {
    // Try to insert an entry into the LocalIdMap while holding an exclusive
    // (write) lock
    LocalIdMap::Accessor accessor;
    auto created = local_id_map->Insert(accessor, id);
    if (unlikely(!created)) {
      // We raced with another writer, use the rep it created and return
      rep = boost::any_cast<irtkReconstructionEbb *>(accessor->second);
    } else {
      // Create a new rep and insert it into the LocalIdMap
      rep = new irtkReconstructionEbb(id);
      accessor->second = rep;
    }
  }
  // Cache the reference to the rep in the local translation table
  EbbRef<irtkReconstructionEbb>::CacheRef(id, *rep);
  return *rep;
}

ebbrt::Future<void> irtkReconstructionEbb::Ping(Messenger::NetworkId nid) {
  uint32_t id;
  Promise<void> promise;
  auto ret = promise.GetFuture();
  {
    std::lock_guard<std::mutex> guard(m_);
    id = id_; // Get a new id (always even)
    id_ += 2;

    bool inserted;
    // insert our promise into the hash table
    std::tie(std::ignore, inserted) =
        promise_map_.emplace(id, std::move(promise));
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

std::unique_ptr<ebbrt::MutUniqueIOBuf> irtkReconstructionEbb::SerializeSlices()
{
    auto buf = MakeUniqueIOBuf(1 * sizeof(int));
    auto dp = buf->GetMutDataPointer();
    dp.Get<int>() = _slices.size();

    for (int j = 0; j < _slices.size(); j++) 
    {
	buf->PrependChain(std::move(serializeImageAttr(_slices[j])));
	buf->PrependChain(std::move(serializeImageI2W(_slices[j])));
	buf->PrependChain(std::move(serializeImageW2I(_slices[j])));
	buf->PrependChain(std::move(serializeSlices(_slices[j])));
    }
    
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> irtkReconstructionEbb::SerializeMask()
{
    auto buf = MakeUniqueIOBuf(0);
    buf->PrependChain(std::move(serializeImageAttr(_mask)));
    buf->PrependChain(std::move(serializeImageI2W(_mask)));
    buf->PrependChain(std::move(serializeImageW2I(_mask)));
    buf->PrependChain(std::move(serializeSlices(_mask)));
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> irtkReconstructionEbb::SerializeReconstructed()
{
    auto buf = MakeUniqueIOBuf(0);
    buf->PrependChain(std::move(serializeImageAttr(_reconstructed)));
    buf->PrependChain(std::move(serializeImageI2W(_reconstructed)));
    buf->PrependChain(std::move(serializeImageW2I(_reconstructed)));
    buf->PrependChain(std::move(serializeSlices(_reconstructed)));
    return buf;
}

std::unique_ptr<ebbrt::MutUniqueIOBuf> irtkReconstructionEbb::SerializeTransformations()
{
    auto buf = MakeUniqueIOBuf(1 * sizeof(int));
    auto dp = buf->GetMutDataPointer();
    dp.Get<int>() = _transformations.size();

    for(int j = 0; j < _transformations.size(); j++)
    {
	buf->PrependChain(std::move(serializeRigidTrans(_transformations[j])));
    }

    return buf;
}

void irtkReconstructionEbb::DeserializeSlice(ebbrt::IOBuf::DataPointer& dp, irtkRealImage& tmp)
{
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

    //FORPRINTF("%d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
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

void irtkReconstructionEbb::DeserializeTransformations(ebbrt::IOBuf::DataPointer& dp, irtkRigidTransformation& tmp)
{
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
	
	//FORPRINTF("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d %d %d %d %d %d %lf\n", tx, ty, tz, rx, ry, rz, cosrx, cosry, cosrz, sinx, siny, sinz, status0, status1, status2, status3, status4, status5, rows, cols, mat.Sum());
	irtkRigidTransformation irt(tx, ty, tz, rx, ry, rz, cosrx, cosry, cosrz, sinx, siny, sinz, status0, status1, status2, status3, status4, status5, mat);

	tmp = std::move(irt);
	//_transformations[i] = std::move(irt);
	//FORPRINTF("%lf %lf\n", _transformations[i].Sum(), _transformations[i]._matrix.Sum());
}

void irtkReconstructionEbb::SendRecon(int iterations) {
    double delta = 150;
    double lastIterLambda = 0.01;
    int rec_iterations_first = 4;
    int rec_iterations_last = 13;
    bool intensity_matching = true;
    double lambda = 0.02;
    int levels = 3;

    RunRecon(iterations, delta, lastIterLambda, rec_iterations_first,
	     rec_iterations_last, intensity_matching, lambda, levels);
}

void irtkReconstructionEbb::addNid(ebbrt::Messenger::NetworkId nid) {
  nids.push_back(nid);
  if ((int)nids.size() == numNodes) {
    nodesinit.SetValue();
  }
}

ebbrt::Future<void> irtkReconstructionEbb::waitPool() {
  return std::move(nodesinit.GetFuture());
}

ebbrt::Future<void> irtkReconstructionEbb::waitReceive() {
  return std::move(mypromise.GetFuture());
}

void bbox(irtkRealImage &stack, irtkRigidTransformation &transformation,
          double &min_x, double &min_y, double &min_z, double &max_x,
          double &max_y, double &max_z) {

  cout << "bbox" << endl;

  min_x = voxel_limits<irtkRealPixel>::max();
  min_y = voxel_limits<irtkRealPixel>::max();
  min_z = voxel_limits<irtkRealPixel>::max();
  max_x = voxel_limits<irtkRealPixel>::min();
  max_y = voxel_limits<irtkRealPixel>::min();
  max_z = voxel_limits<irtkRealPixel>::min();
  double x, y, z;
  for (int i = 0; i <= stack.GetX(); i += stack.GetX())
    for (int j = 0; j <= stack.GetY(); j += stack.GetY())
      for (int k = 0; k <= stack.GetZ(); k += stack.GetZ()) {
        x = i;
        y = j;
        z = k;
        stack.ImageToWorld(x, y, z);
        // [fetalReconstruction] FIXME!!!
        transformation.Transform(x, y, z);
        //  [fetalReconstruction] transformation.Inverse( x, y, z );
        if (x < min_x)
          min_x = x;
        if (y < min_y)
          min_y = y;
        if (z < min_z)
          min_z = z;
        if (x > max_x)
          max_x = x;
        if (y > max_y)
          max_y = y;
        if (z > max_z)
          max_z = z;
      }
}

void bboxCrop(irtkRealImage &image) {
  int min_x, min_y, min_z, max_x, max_y, max_z;
  min_x = image.GetX() - 1;
  min_y = image.GetY() - 1;
  min_z = image.GetZ() - 1;
  max_x = 0;
  max_y = 0;
  max_z = 0;
  for (int i = 0; i < image.GetX(); i++)
    for (int j = 0; j < image.GetY(); j++)
      for (int k = 0; k < image.GetZ(); k++) {
        if (image.Get(i, j, k) > 0) {
          if (i < min_x)
            min_x = i;
          if (j < min_y)
            min_y = j;
          if (k < min_z)
            min_z = k;
          if (i > max_x)
            max_x = i;
          if (j > max_y)
            max_y = j;
          if (k > max_z)
            max_z = k;
        }
      }

  //[fetalReconstruction] Cut region of interest
  image = image.GetRegion(min_x, min_y, min_z, max_x + 1, max_y + 1, max_z + 1);
}

void centroid(irtkRealImage &image, double &x, double &y, double &z) {
  double sum_x = 0;
  double sum_y = 0;
  double sum_z = 0;
  double norm = 0;
  double v;
  for (int i = 0; i < image.GetX(); i++)
    for (int j = 0; j < image.GetY(); j++)
      for (int k = 0; k < image.GetZ(); k++) {
        v = image.Get(i, j, k);
        if (v <= 0)
          continue;
        sum_x += v * i;
        sum_y += v * j;
        sum_z += v * k;
        norm += v;
      }

  x = sum_x / norm;
  y = sum_y / norm;
  z = sum_z / norm;

  image.ImageToWorld(x, y, z);
}

irtkReconstructionEbb::~irtkReconstructionEbb() {}

void irtkReconstructionEbb::CenterStacks(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations,
    int templateNumber) {
  // [fetalReconstruction] template center
  double x0, y0, z0;
  irtkRealImage mask;
  mask = stacks[templateNumber] != -1;
  centroid(mask, x0, y0, z0);

  double x, y, z;
  irtkMatrix m1, m2;
  for (unsigned int i = 0; i < stacks.size(); i++) {
    if (i == templateNumber)
      continue;

    mask = stacks[i] != -1;
    centroid(mask, x, y, z);

    irtkRigidTransformation translation;
    translation.PutTranslationX(x0 - x);
    translation.PutTranslationY(y0 - y);
    translation.PutTranslationZ(z0 - z);

    m1 = stack_transformations[i].GetMatrix();
    m2 = translation.GetMatrix();
    stack_transformations[i].PutMatrix(m2 * m1);
  }
}

class ParallelAverage {
  irtkReconstructionEbb *reconstructor;
  vector<irtkRealImage> &stacks;
  vector<irtkRigidTransformation> &stack_transformations;

  ///[fetalReconstruction] Padding value in target (voxels in the target image with this
  ///[fetalReconstruction] value will be ignored)
  double targetPadding;

  ///[fetalReconstruction] Padding value in source (voxels outside the source image will
  ///[fetalReconstruction] be set to this value)
  double sourcePadding;

  double background;

  //[fetalReconstruction] Volumetric registrations are stack-to-template while slice-to-volume
  //[fetalReconstruction] registrations are actually performed as volume-to-slice
  //[fetalReconstruction] (reasons: technicalities of implementation)
  //[fetalReconstruction] so transformations need to be inverted beforehand.

  bool linear;

  int nt;

public:
  irtkRealImage average;
  irtkRealImage weights;

  void operator()(const blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      irtkImageTransformation imagetransformation;
      irtkImageFunction *interpolator;
      if (linear)
        interpolator = new irtkLinearInterpolateImageFunction;
      else
        interpolator = new irtkNearestNeighborInterpolateImageFunction;

      irtkRealImage s = stacks[i];
      irtkRigidTransformation t = stack_transformations[i];
      imagetransformation.SetInput(&s, &t);
      irtkRealImage image(reconstructor->_reconstructed.GetImageAttributes());
      image = 0;

      imagetransformation.SetOutput(&image);
      imagetransformation.PutTargetPaddingValue(targetPadding);
      imagetransformation.PutSourcePaddingValue(sourcePadding);
      imagetransformation.PutInterpolator(interpolator);
      imagetransformation.Run();

      irtkRealPixel *pa = average.GetPointerToVoxels();
      irtkRealPixel *pi = image.GetPointerToVoxels();
      irtkRealPixel *pw = weights.GetPointerToVoxels();
      for (int p = 0; p < average.GetNumberOfVoxels(); p++) {
        if (*pi != background) {
          *pa += *pi;
          *pw += 1;
        }
        pa++;
        pi++;
        pw++;
      }
      delete interpolator;
    }
  }

  ParallelAverage(ParallelAverage &x, split)
      : reconstructor(x.reconstructor), stacks(x.stacks),
        stack_transformations(x.stack_transformations) {
    average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    average = 0;
    weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    weights = 0;
    targetPadding = x.targetPadding;
    sourcePadding = x.sourcePadding;
    background = x.background;
    linear = x.linear;
  }

  void join(const ParallelAverage &y) {
    average += y.average;
    weights += y.weights;
  }

  ParallelAverage(irtkReconstructionEbb *reconstructor,
                  vector<irtkRealImage> &_stacks,
                  vector<irtkRigidTransformation> &_stack_transformations,
                  double _targetPadding, double _sourcePadding,
                  double _background, int _nt, bool _linear = false)
      : reconstructor(reconstructor), stacks(_stacks),
        stack_transformations(_stack_transformations) {
    average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    average = 0;
    weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    weights = 0;
    targetPadding = _targetPadding;
    sourcePadding = _sourcePadding;
    background = _background;
    linear = _linear;
    nt = _nt;
  }

  void operator()() {
    task_scheduler_init init(nt);
    parallel_reduce(blocked_range<size_t>(0, stacks.size()), *this);
    init.terminate();
  }
};

irtkRealImage irtkReconstructionEbb::CreateAverage(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations) {
  if (!_template_created) {
    cerr << "Please create the template before calculating the average of the "
            "stacks."
         << endl;
    exit(1);
  }

  InvertStackTransformations(stack_transformations);
  ParallelAverage parallelAverage(this, stacks, stack_transformations, -1, 0,
                                  0, // [fetalRecontruction] target/source/background
                                  true, _numThreads);
  parallelAverage();
  irtkRealImage average = parallelAverage.average;
  irtkRealImage weights = parallelAverage.weights;
  average /= weights;
  InvertStackTransformations(stack_transformations);
  return average;
}

double irtkReconstructionEbb::CreateTemplate(irtkRealImage stack,
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

  _reconstructed_gpu.Initialize(_reconstructed.GetImageAttributes());
  _template_created = true;

  // [fetalRecontruction] return resulting resolution of the template image
  return d;
}

irtkRealImage irtkReconstructionEbb::CreateMask(irtkRealImage image) {
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

void irtkReconstructionEbb::SetMask(irtkRealImage *mask, double sigma,
                                    double threshold) {
  if (!_template_created) {
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
  _have_mask = true;

  if (_debug)
    _mask.Write("mask.nii");
}

void irtkReconstructionEbb::TransformMask(
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

void irtkReconstructionEbb::ResetOrigin(
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

void irtkReconstructionEbb::ResetOrigin(
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

class ParallelStackRegistrations {
  irtkReconstructionEbb *reconstructor;
  vector<irtkRealImage> &stacks;
  vector<irtkRigidTransformation> &stack_transformations;
  int templateNumber;
  irtkGreyImage &target;
  irtkRigidTransformation &offset;
  bool _externalTemplate;
  int nt;

public:
  ParallelStackRegistrations(
      irtkReconstructionEbb *_reconstructor, vector<irtkRealImage> &_stacks,
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

void irtkReconstructionEbb::StackRegistrations(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations, int templateNumber,
    bool useExternalTarget) {
  
  InvertStackTransformations(stack_transformations);

  // [fetalRecontruction] template is set as the target
  irtkGreyImage target;
  if (!useExternalTarget) {
    target = stacks[templateNumber];
  } else {
    target = externalRegistrationTargetImage;
  }

  // [fetalRecontruction] target needs to be masked before registration
  if (_have_mask) {
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
  ParallelStackRegistrations registration(this, stacks, stack_transformations,
                                          templateNumber, target, offset,
                                          _numThreads, useExternalTarget);
  registration();
  InvertStackTransformations(stack_transformations);
}

void irtkReconstructionEbb::RestoreSliceIntensities() {
  unsigned int inputIndex;
  int i;
  double factor;
  irtkRealPixel *p;

#ifndef _EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf(1 * sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 11;
      //FORPRINTF("RestoreSliceIntensities : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
#endif

  //FORPRINTF("sumPartImage: %lf\n", sumPartImage(_slices, _start, _end));
  
  for (inputIndex = _start; inputIndex < _end; inputIndex++)
  {
      // [fetalRecontruction] calculate scaling factor 
      // [fetalRecontruction] _average_value;
      factor = _stack_factor[_stack_index[inputIndex]];
      // [fetalRecontruction] read the pointer to current slice
      p = _slices[inputIndex].GetPointerToVoxels();
      for (i = 0; i < _slices[inputIndex].GetNumberOfVoxels(); i++)
      {
	  if (*p > 0)
	      *p = *p / factor;
	  p++;
      }
  }

  //FORPRINTF("sumPartImage: %lf\n", sumPartImage(_slices, _start, _end));
#ifndef __EBBRT_BM__
  if (_debug) 
    FORPRINTF("[H] RestoreIntensities: Bloacking \n");
  testFuture = ebbrt::Promise<int>();
  auto tf = testFuture.GetFuture();
  tf.Block();
  if (_debug) 
    FORPRINTF("[H] RestoreIntensities: Returning from future \n");
#endif
  
}

void irtkReconstructionEbb::ScaleVolume() {
  unsigned int inputIndex;
  int i, j;
  _sscalenum = 0;
  _sscaleden = 0;

#ifndef __EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf(1 * sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 12;
      bytesTotal += buf->ComputeChainDataLength();
      //FORPRINTF("ScaleVolume : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      SendMessage(nids[i], std::move(buf));
  }
#endif

  //FORPRINTF("_weights = %lf\n_slices = %lf\n_simulated_slices = %lf\n_slice_weight_cpu = %lf\n", sumPartImage(_weights, _start, _end), sumPartImage(_slices, _start, _end), sumPartImage(_simulated_slices, _start, _end), sumPartVec(_slice_weight_cpu, _start, _end) );

  
  for (inputIndex = _start; inputIndex < _end; inputIndex++)
  {
    // [fetalRecontruction] alias for the current slice
    irtkRealImage &slice = _slices[inputIndex];

    // [fetalRecontruction] alias for the current weight image
    irtkRealImage &w = _weights[inputIndex];

    // [fetalRecontruction] alias for the current simulated slice
    irtkRealImage &sim = _simulated_slices[inputIndex];

    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] scale - intensity matching
          if (_simulated_weights[inputIndex](i, j, 0) > 0.99)
	  {
            _sscalenum += w(i, j, 0) * _slice_weight_cpu[inputIndex] *
                        slice(i, j, 0) * sim(i, j, 0);
            _sscaleden += w(i, j, 0) * _slice_weight_cpu[inputIndex] *
                        sim(i, j, 0) * sim(i, j, 0);
          }
        }
  } // [fetalRecontruction] end of loop for a slice inputIndex

  //FORPRINTF("_sscalenum = %lf, _sscaleden = %lf\n", _sscalenum, _sscaleden);
    
#ifdef __EBBRT_BM__
  return;
#endif

#ifndef __EBBRT_BM__
  if (_debug) 
    FORPRINTF("[H] ScaleVolume: Bloacking \n");
  gaussianreconFuture = ebbrt::Promise<int>();
  auto tf = gaussianreconFuture.GetFuture();
  tf.Block();
  if (_debug) 
    FORPRINTF("[H] ScaleVolume: Returning from future \n");
#endif
  
  // [fetalRecontruction] calculate scale for the volume
  double scale = _sscalenum / _sscaleden;
  //FORPRINTF("scale = %lf, recon = %lf\n", scale, _reconstructed.Sum());
  
  irtkRealPixel *ptr = _reconstructed.GetPointerToVoxels();
  for (i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*ptr > 0)
      *ptr = *ptr * scale;
    ptr++;
  }
}

void runSimulateSlices(irtkReconstructionEbb *reconstructor, int start,
                       int end) {
  for (int inputIndex = start; inputIndex != end; ++inputIndex) {
/*#ifdef __EBBRT_BM__
      FORPRINTF("%d\n", inputIndex);
#endif*/
    // [fetalRecontruction] Calculate simulated slice
    reconstructor->_simulated_slices[inputIndex].Initialize(
        reconstructor->_slices[inputIndex].GetImageAttributes());

    reconstructor->_simulated_slices[inputIndex] = 0;

    reconstructor->_simulated_weights[inputIndex].Initialize(
        reconstructor->_slices[inputIndex].GetImageAttributes());

    reconstructor->_simulated_weights[inputIndex] = 0;

    reconstructor->_simulated_inside[inputIndex].Initialize(
        reconstructor->_slices[inputIndex].GetImageAttributes());
    
    reconstructor->_simulated_inside[inputIndex] = 0;
    reconstructor->_slice_inside_cpu[inputIndex] = 0;

    POINT3D p;
    for (unsigned int i = 0; i < reconstructor->_slices[inputIndex].GetX();
         i++) {
      for (unsigned int j = 0; j < reconstructor->_slices[inputIndex].GetY();
           j++) {
        if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
          double weight = 0;
          int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

          for (unsigned int k = 0; k < n; k++) {
            p = reconstructor->_volcoeffs[inputIndex][i][j][k];

            reconstructor->_simulated_slices[inputIndex](i, j, 0) +=
                p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
            weight += p.value;

            if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
              reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
              reconstructor->_slice_inside_cpu[inputIndex] = 1;
            }
          }

          if (weight > 0) {
            reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
            reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;
          }
        }
      }
    }
  }
}

class ParallelSimulateSlices {
  irtkReconstructionEbb *reconstructor;
  int nt;
  int start;
  int end;
public:
    ParallelSimulateSlices(irtkReconstructionEbb *_reconstructor, int _nt, int _start, int _end)
	: reconstructor(_reconstructor), nt(_nt), start(_start), end(_end) {}

  void operator()(const blocked_range<int> &r) const {
    runSimulateSlices(reconstructor, r.begin(), r.end());
  }

  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<int>(0, (end-start)), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::SimulateSlices(bool a) {
#ifndef __EBBRT_BM__
    for (int i = 0; i < (int)nids.size(); i++) 
    {
	auto buf = MakeUniqueIOBuf(1 * sizeof(int));
	auto dp = buf->GetMutDataPointer();
	if(a) dp.Get<int>() = 9;
	else dp.Get<int>() = 2;
	buf->PrependChain(std::move(serializeSlices(_reconstructed)));
    if (_debug) 
	  FORPRINTF("[H] SimulateSlices : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	bytesTotal += buf->ComputeChainDataLength();
	SendMessage(nids[i], std::move(buf));
    }
    ParallelSimulateSlices parallelSimulateSlices(this, _numThreads, _start, _end);
    parallelSimulateSlices();

#else
    //FORPRINTF("simulateslices start\n");
    size_t ncpus = ebbrt::Cpu::Count();
    static ebbrt::SpinBarrier bar(ncpus);
    ebbrt::EventManager::EventContext context;
    std::atomic<size_t> count(0);
    size_t theCpu = ebbrt::Cpu::GetMine();
    int diff = (_end-_start);
    for (size_t i = 0; i < ncpus; i++) {
      // spawn jobs on each core using SpawnRemote
      ebbrt::event_manager->SpawnRemote(
          [this, theCpu, ncpus, &count, &context, i, diff]() {
            // get my cpu id
            size_t mycpu = ebbrt::Cpu::GetMine();
            int starte, ende, factor;
            factor = (int)ceil(diff / (float)ncpus);
	    starte = (i * factor) + _start;
	    ende = (i * factor + factor) + _start;
            ende = (ende > _end) ? _end : ende;
            runSimulateSlices(this, starte, ende);
            count++;
        if (_debug) 
	      FORPRINTF("[BM] SimulateSlices CPU %d barrier wait\n", mycpu);
            bar.Wait();
        if (_debug) { 
	      FORPRINTF("[BM] SimulateSlices CPU %d barrier done\n", mycpu);
	      FORPRINTF("[BM] SimulateSlices CPU %d while wait\n", mycpu);
        }
            while (count < (size_t)ncpus)
              ;
        if (_debug) 
	      FORPRINTF("[BM] SimulateSlices CPU %d while done\n", mycpu);
            if (mycpu == theCpu) {
              ebbrt::event_manager->ActivateContext(std::move(context));
            }
          },
          indexToCPU(
              i)); // if i don't add indexToCPU, one of the cores never run ? ?
    }
    ebbrt::event_manager->SaveContext(context);
    //FORPRINTF("runSimulateSlices\n");
    //runSimulateSlices(this, _start, _end);
#endif

/*FORPRINTF("\n****** SimulateSlices*****\n");
  FORPRINTF("\n_start = %d, _end = %d, \t_simulated_slices=%lf\n\tsimulated_weights=%lf\n\tsimulated_inside=%lf\n\t_slice_inside_cpu=%d\n\n", _start, _end, sumPartImage(_simulated_slices, _start, _end), sumPartImage(_simulated_weights, _start, _end), sumPartImage(_simulated_inside, _start, _end), sumPartInt(_slice_inside_cpu, _start, _end));*/
//FORPRINTF("\n\t_simulated_slices=%lf\n\tsimulated_weights=%lf\n\tsimulated_inside=%lf\n\t_slice_inside_cpu=%d\n\n", sumImage(_simulated_slices), sumImage(_simulated_weights), sumImage(_simulated_inside), sumInt(_slice_inside_cpu));
//FORPRINTF("\n**********************\n");

#ifndef __EBBRT_BM__
if (_debug)
  FORPRINTF("[H] SimulateSlices: Bloacking \n");
testFuture = ebbrt::Promise<int>();
auto tf = testFuture.GetFuture();
tf.Block();
if (_debug)
  FORPRINTF("[H] SimulateSlices: Returning from future \n");
#endif
}

void irtkReconstructionEbb::SimulateStacks(vector<irtkRealImage> &stacks) {
  unsigned int inputIndex;
  int i, j, k, n;
  irtkRealImage sim;
  POINT3D p;
  double weight;

  int z, current_stack;
  z = -1;             // [fetalRecontruction] this is the z coordinate of the stack
  current_stack = -1; // [fetalRecontruction] need to know when to start new stack

  for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

    // [fetalRecontruction] read the current slice
    irtkRealImage &slice = _slices[inputIndex];

    // [fetalRecontruction] Calculate simulated slice
    sim.Initialize(slice.GetImageAttributes());
    sim = 0;

    // [fetalRecontruction] do not simulate excluded slice
    if (_slice_weight_cpu[inputIndex] > 0.5) {
      for (i = 0; i < slice.GetX(); i++)
        for (j = 0; j < slice.GetY(); j++)
          if (slice(i, j, 0) != -1) {
            weight = 0;
            n = _volcoeffs[inputIndex][i][j].size();
            for (k = 0; k < n; k++) {
              p = _volcoeffs[inputIndex][i][j][k];
              sim(i, j, 0) += p.value * _reconstructed(p.x, p.y, p.z);
              weight += p.value;
            }
            if (weight > 0)
              sim(i, j, 0) /= weight;
          }
    }

    if (_stack_index[inputIndex] == current_stack)
      z++;
    else {
      current_stack = _stack_index[inputIndex];
      z = 0;
    }

    for (i = 0; i < sim.GetX(); i++)
      for (j = 0; j < sim.GetY(); j++) {
        stacks[_stack_index[inputIndex]](i, j, z) = sim(i, j, 0);
      }
    // [fetalRecontruction] end of loop for a slice inputIndex
  }
}

void irtkReconstructionEbb::MatchStackIntensities(
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

  // [fetalRecontruction] remember the set average value
  _average_value = averageValue;

  // [fetalRecontruction] averages need to be calculated only in ROI
  for (ind = 0; ind < stacks.size(); ind++) {
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
          if (stacks[ind](i, j, k) > 0) {
            sum += stacks[ind](i, j, k);
            num++;
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
      _stack_factor.push_back(factor);
    } else {
      factor = averageValue / stack_average[ind];
      _stack_factor.push_back(factor);
    }

    ptr = stacks[ind].GetPointerToVoxels();
    for (i = 0; i < stacks[ind].GetNumberOfVoxels(); i++) {
      if (*ptr > 0)
        *ptr *= factor;
      ptr++;
    }
  }
}

void irtkReconstructionEbb::MatchStackIntensitiesWithMasking(
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
  _average_value = averageValue;

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
      _stack_factor.push_back(factor);
    } else {
      factor = averageValue / stack_average[ind];
      _stack_factor.push_back(factor);
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
      cout << _stack_factor[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }
}

void irtkReconstructionEbb::generatePSFVolume() {
  double dx, dy, dz;
  // [fetalRecontruction] currentlz just a test function
  _slices[_slices.size() - 1].GetPixelSize(&dx, &dy, &dz);

  // [fetalRecontruction] sigma of 3D Gaussian 
  // [fetalRecontruction] (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM
  // [fetalRecontruction] = dz through-plane)
  double psfSigma = 2.3548;

  double size = _reconstructed.GetXSize() / _quality_factor;

  // [fetalRecontruction] TODO: TUne until same as CPU
  // [fetalRecontruction] number of voxels in each direction
  // [fetalRecontruction] the ROI is 2*voxel dimension

  int xDim = PSF_SIZE; 
  int yDim = PSF_SIZE;
  int zDim = PSF_SIZE;

  double ldim;
  ldim = xDim;
  ldim = (ldim > yDim) ? ldim : yDim;
  ldim = (ldim > zDim) ? ldim : zDim;
  double xsize = xDim / ldim;
  double ysize = yDim / ldim;
  double zsize = zDim / ldim;

  double sigmax = (1.2) * (dx) / (psfSigma);
  double sigmay = (1.2) * (dy) / (psfSigma);
  double sigmaz = (1.0) * (dz) / (psfSigma);

  // [fetalRecontruction] image corresponding to PSF
  irtkImageAttributes attr;
  attr._x = xDim;
  attr._y = yDim;
  attr._z = zDim;
  attr._dx = _reconstructed.GetXSize();
  attr._dy = _reconstructed.GetYSize();
  attr._dz = _reconstructed.GetZSize();
  irtkGenericImage<float> PSF(attr);

  // [fetalRecontruction] centre of PSF
  double cx, cy, cz;
  cx = 0.5 * (attr._x - 1);
  cy = 0.5 * (attr._y - 1);
  cz = 0.5 * (attr._z - 1);
  PSF.ImageToWorld(cx, cy, cz);

  double x, y, z;
  double sum = 0;
  int i, j, k;
  for (i = 0; i < attr._x; i++)
    for (j = 0; j < attr._y; j++)
      for (k = 0; k < attr._z; k++) {
        x = i;
        y = j;
        z = k;

        PSF.ImageToWorld(x, y, z);

        x -= cx;
        y -= cy;
        z -= cz;

        // [fetalRecontruction] sinc gauss
        double R =
            sqrt(x * x / (2 * sigmax * sigmax) + y * y / (2 * sigmay * sigmay) +
                 z * z / (2 * sigmaz * sigmaz));
        PSF(i, j, k) = sin(R) / (R)*exp(-x * x / (2 * sigmax * sigmax) -
                                        y * y / (2 * sigmay * sigmay) -
                                        z * z / (2 * sigmaz * sigmaz));
        sum += PSF(i, j, k);
      }
  PSF /= sum;
}

void irtkReconstructionEbb::CreateSlicesAndTransformations(
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
      _simulated_slices.push_back(slice);
      _simulated_weights.push_back(slice);
      _simulated_inside.push_back(slice);
      // [fetalRecontruction] remeber stack index for this slice
      _stack_index.push_back(i);
      // [fetalRecontruction] initialize slice transformation with 
      // [fetalRecontruction] the stack transformation
      _transformations.push_back(stack_transformations[i]);
      _transformations_gpu.push_back(stack_transformations[i]);
    }
  }
}

void irtkReconstructionEbb::ResetSlices(vector<irtkRealImage> &stacks,
                                        vector<double> &thickness) {
  _slices.clear();

  // [fetalRecontruction] for each stack
  for (unsigned int i = 0; i < stacks.size(); i++) {
    // [fetalRecontruction] image attributes contain image and voxel size
    irtkImageAttributes attr = stacks[i].GetImageAttributes();

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
    }
  }
  for (int i = 0; i < _slices.size(); i++) {
    _bias[i].Initialize(_slices[i].GetImageAttributes());
    _weights[i].Initialize(_slices[i].GetImageAttributes());
  }
}

void irtkReconstructionEbb::SetSlicesAndTransformations(
    vector<irtkRealImage> &slices,
    vector<irtkRigidTransformation> &slice_transformations,
    vector<int> &stack_ids, vector<double> &thickness) {
  _slices.clear();
  _stack_index.clear();
  _transformations.clear();
  _transformations_gpu.clear();
  _slices.clear();
  _simulated_slices.clear();
  _simulated_weights.clear();
  _simulated_inside.clear();

  for (unsigned int i = 0; i < slices.size(); i++) {
    irtkRealImage slice = slices[i];
    // [fetalRecontruction] set correct voxel size in the stack. 
    // [fetalRecontruction] Z size is equal to slice thickness.
    slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), thickness[i]);
    // [fetalRecontruction] remember the slice
    _slices.push_back(slice);
    _simulated_slices.push_back(slice);
    _simulated_weights.push_back(slice);
    _simulated_inside.push_back(slice);
    // [fetalRecontruction] remember stack index for this slice
    _stack_index.push_back(stack_ids[i]);
    // [fetalRecontruction] get slice transformation
    _transformations.push_back(slice_transformations[i]);
    _transformations_gpu.push_back(slice_transformations[i]);
  }
}

void irtkReconstructionEbb::UpdateSlices(vector<irtkRealImage> &stacks,
                                         vector<double> &thickness) {
  _slices.clear();
  for (unsigned int i = 0; i < stacks.size(); i++) {
    irtkImageAttributes attr = stacks[i].GetImageAttributes();

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
    }
  }
}

void irtkReconstructionEbb::MaskSlices() {

  double x, y, z;
  int i, j;

  // [fetalRecontruction] Check whether we have a mask
  if (!_have_mask) {
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

void runSliceToVolumeRegistration(irtkReconstructionEbb *reconstructor,
                                  int start, int end) {
  irtkImageAttributes attr = reconstructor->_reconstructed.GetImageAttributes();

  for (int inputIndex = start; inputIndex != end; inputIndex++) {
    irtkImageRigidRegistrationWithPadding registration;
    irtkGreyPixel smin, smax;
    irtkGreyImage target;
    irtkRealImage slice, w, b, t;
    irtkResamplingWithPadding<irtkRealPixel> resampling(attr._dx, attr._dx,
                                                        attr._dx, -1);

    t = reconstructor->_slices[inputIndex];
    resampling.SetInput(&reconstructor->_slices[inputIndex]);
    resampling.SetOutput(&t);
    resampling.Run();
    target = t;
    target.GetMinMax(&smin, &smax);

    if (smax > -1) {
      // [fetalRecontruction] put origin to zero
      irtkRigidTransformation offset;
      reconstructor->ResetOrigin(target, offset);
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = reconstructor->_transformations[inputIndex].GetMatrix();
      m = m * mo;
      reconstructor->_transformations[inputIndex].PutMatrix(m);

      irtkGreyImage source = reconstructor->_reconstructed;
      registration.SetInput(&target, &source);
      registration.SetOutput(&reconstructor->_transformations[inputIndex]);
      registration.GuessParameterSliceToVolume();
      registration.SetTargetPadding(-1);
      registration.Run();

      reconstructor->_slices_regCertainty[inputIndex] =
          registration.last_similarity;
      // [fetalRecontruction] undo the offset
      mo.Invert();
      m = reconstructor->_transformations[inputIndex].GetMatrix();
      m = m * mo;
      reconstructor->_transformations[inputIndex].PutMatrix(m);
    }
  }
}

class ParallelSliceToVolumeRegistration {
public:
    irtkReconstructionEbb *reconstructor;
    int nt;
    int start;
    int end;
    
    ParallelSliceToVolumeRegistration(irtkReconstructionEbb *_reconstructor,
                                    int _nt, int _start, int _end)
      : reconstructor(_reconstructor), nt(_nt), start(_start), end(_end) {}


  void operator()(const blocked_range<int> &r) const {
    runSliceToVolumeRegistration(reconstructor, r.begin(), r.end());
  }

  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<int>(0, (end-start)), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::SliceToVolumeRegistration() {
  if (_slices_regCertainty.size() == 0) {
    _slices_regCertainty.resize(_slices.size());
  }

#ifndef __EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf((1 * sizeof(int)));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 13;
      buf->PrependChain(std::move(serializeSlices(_reconstructed)));
      
      //FORPRINTF("SliceToVolumeRegistration : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
  
  ParallelSliceToVolumeRegistration registration(this, _numThreads, _start, _end);
  registration();
  
  if (_debug)
    FORPRINTF("[H] SliceToVolume: Bloacking \n");
  testFuture = ebbrt::Promise<int>();
  auto tf = testFuture.GetFuture();
  tf.Block();
  if (_debug)
    FORPRINTF("[H] SliceToVolume: Returning from future\n");
  
  //FORPRINTF("SliceToVolumeRegistration : %lf %lf %lf %lf\n", sumTrans(_transformations, 0, _slices.size()), sumTrans2(_transformations, 0, _slices.size()), sumPartImage(_slices, 0, _slices.size()), _reconstructed.Sum());
#else
  size_t ncpus = ebbrt::Cpu::Count();
  static ebbrt::SpinBarrier bar(ncpus);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t theCpu = ebbrt::Cpu::GetMine();
  int diff = (_end-_start);
  for (size_t i = 0; i < ncpus; i++) {
      // spawn jobs on each core using SpawnRemote
      ebbrt::event_manager->SpawnRemote(
          [this, theCpu, ncpus, &count, &context, i, diff]() {
	      // get my cpu id
	      size_t mycpu = ebbrt::Cpu::GetMine();
	      int starte, ende, factor;
	      factor = (int)ceil(diff / (float)ncpus);
	      starte = (i * factor) + _start;
	      ende = (i * factor + factor) + _start;
	      ende = (ende > _end) ? _end : ende;
	      runSliceToVolumeRegistration(this, starte, ende);
	      count++;
          if (_debug)
            FORPRINTF("[BM] ParallelSliceToVolume CPU %d barrier wait\n", mycpu);
	      bar.Wait();
          if (_debug) {
            FORPRINTF("[BM] ParallelSliceToVolume CPU %d barrier done\n", mycpu);
	        FORPRINTF("[BM] ParallelSliceToVolume CPU %d while wait\n", mycpu);
          }
	      while (count < (size_t)ncpus)
		  ;
          if (_debug)
            FORPRINTF("[BM] ParallelSliceToVolume CPU %d while done\n", mycpu);
	      if (mycpu == theCpu) {
		  ebbrt::event_manager->ActivateContext(std::move(context));
	      }
          },
          indexToCPU(
              i)); // if i don't add indexToCPU, one of the cores never run ? ?
  }
  ebbrt::event_manager->SaveContext(context);
  
  //FORPRINTF("runSlicesToVolumeRegistration\n");
  //runSliceToVolumeRegistration(this, _start, _end);
  //FORPRINTF("SliceToVolumeRegistration : %lf %lf %lf\n", sumTrans(_transformations, _start, _end), sumPartImage(_slices, _start, _end), _reconstructed.Sum());
#endif
}

void irtkReconstructionEbb::InitVoxelStruct() {
  int i, j, k;
  auto reconAttr = _reconstructed.GetImageAttributes();
  //FORPRINTF(
  //    "irtkReconstructionEbb::InitVoxelStruct - _x = %d, _y = %d, _z = %d\n",
   //   reconAttr._x, reconAttr._y, reconAttr._z);

  _invertvolcoeffs.resize(reconAttr._x);
  for (i = 0; i < reconAttr._x; i++) {
    _invertvolcoeffs[i].resize(reconAttr._y);
    for (j = 0; j < reconAttr._y; j++) {
      _invertvolcoeffs[i][j].resize(reconAttr._z);
      for (k = 0; k < reconAttr._z; k++) {
        _invertvolcoeffs[i][j][k].resize(_slices.size());
      }
    }
  }
}

void runCoeffInit(irtkReconstructionEbb *reconstructor, int start, int end) {
    
    for (int inputIndex = start; inputIndex != end; ++inputIndex) {	
      int slice_inside;      

    // [fetalRecontruction] get resolution of the volume
    double vx, vy, vz;
    reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
    // [fetalRecontruction] volume is always isotropic
    double res = vx;

    // [fetalRecontruction] start of a loop for a slice inputIndex
    // [fetalRecontruction] read the slice
    irtkRealImage &slice = reconstructor->_slices[inputIndex];

    // [fetalRecontruction] voxel slice info
    SLICEINFO sinfo;

    // [fetalRecontruction] prepare structures for storage
    POINT3D p;
    VOXELCOEFFS empty;
    SLICECOEFFS slicecoeffs(slice.GetX(),
                            vector<VOXELCOEFFS>(slice.GetY(), empty));

    // [fetalRecontruction] to check whether the slice has an overlap with mask ROI
    slice_inside = 0;

    // [fetalRecontruction] PSF will be calculated in slice space in 
    // [fetalRecontruction] higher resolution

    // [fetalRecontruction] get slice voxel size to define PSF
    double dx, dy, dz;
    slice.GetPixelSize(&dx, &dy, &dz);

    // [fetalRecontruction] sigma of 3D Gaussian (sinc with FWHM=dx or dy 
    // [fetalRecontruction] in-plane, Gaussian with FWHM = dz through-plane)
    double sigmax = 1.2 * dx / 2.3548;
    double sigmay = 1.2 * dy / 2.3548;
    double sigmaz = dz / 2.3548;

    // [fetalRecontruction] calculate discretized PSF

    // i[fetalRecontruction] sotropic voxel size of PSF - derived from 
    // [fetalRecontruction] resolution of reconstructed volume
    double size = res / reconstructor->_quality_factor;

    // [fetalRecontruction] number of voxels in each direction
    // [fetalRecontruction] the ROI is 2*voxel dimension

    int xDim = round(2 * dx / size);
    int yDim = round(2 * dy / size);
    int zDim = round(2 * dz / size);

    // [fetalRecontruction] image corresponding to PSF
    irtkImageAttributes attr;
    attr._x = xDim;
    attr._y = yDim;
    attr._z = zDim;
    attr._dx = size;
    attr._dy = size;
    attr._dz = size;
    irtkRealImage PSF(attr);

    // [fetalRecontruction] centre of PSF
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
          // [fetalRecontruction] continuous PSF does not need to be 
          // [fetalRecontruction] normalized as discrete will be
          PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) -
                             y * y / (2 * sigmay * sigmay) -
                             z * z / (2 * sigmaz * sigmaz));
          sum += PSF(i, j, k);
        }
    PSF /= sum;
    
    // [fetalRecontruction] prepare storage for PSF transformed and resampled 
    // [fetalRecontruction] to the space of reconstructed volume
    // [fetalRecontruction] maximum dim of rotated kernel - the 
    // [fetalRecontruction] next higher odd integer plus two to
    // [fetalRecontruction] accound for rounding error of tx,ty,tz.
    // [fetalRecontruction] Note conversion from PSF image coordinates to tPSF 
    // [fetalRecontruction] image coordinates *size/res
    int dim =
        (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) *
                    size / res) /
               2)) *
            2 +
        1 + 2;
    // [fetalRecontruction] prepare image attributes. Voxel dimension will be 
    // [fetalRecontruction] taken from the reconstructed volume
    attr._x = dim;
    attr._y = dim;
    attr._z = dim;
    attr._dx = res;
    attr._dy = res;
    attr._dz = res;
    // [fetalRecontruction] create matrix from transformed PSF
    irtkRealImage tPSF(attr);
    // [fetalRecontruction] calculate centre of tPSF in image coordinates
    int centre = (dim - 1) / 2;

    // [fetalRecontruction] for each voxel in current slice calculate matrix 
    // [fetalRecontruction] coefficients
    int ii, jj, kk;
    int tx, ty, tz;
    int nx, ny, nz;
    int l, m, n;
    double weight;
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] calculate centrepoint of slice voxel in 
          // [fetalRecontruction] volume space (tx,ty,tz)
          x = i;
          y = j;
          z = 0;
          slice.ImageToWorld(x, y, z);
          reconstructor->_transformations[inputIndex].Transform(x, y, z);
          reconstructor->_reconstructed.WorldToImage(x, y, z);
          tx = round(x);
          ty = round(y);
          tz = round(z);

          // [fetalRecontruction] Clear the transformed PSF
          for (ii = 0; ii < dim; ii++)
            for (jj = 0; jj < dim; jj++)
              for (kk = 0; kk < dim; kk++)
                tPSF(ii, jj, kk) = 0;

          // [fetalRecontruction] for each POINT3D of the PSF
          for (ii = 0; ii < xDim; ii++)
            for (jj = 0; jj < yDim; jj++)
              for (kk = 0; kk < zDim; kk++) {
		  // [fetalRecontruction] Calculate the position of the POINT3D of
                // [fetalRecontruction] PSF centered over current slice voxel
                // [fetalRecontruction] This is a bit complicated because slices
                // [fetalRecontruction] can be oriented in any direction

                // [fetalRecontruction] PSF image coordinates
                x = ii;
                y = jj;
                z = kk;
                // [fetalRecontruction] change to PSF world coordinates 
                // [fetalRecontruction] - now real sizes in mm
                PSF.ImageToWorld(x, y, z);
                // [fetalRecontruction] centre around the centrepoint of the PSF
                x -= cx;
                y -= cy;
                z -= cz;

                // [fetalRecontruction] Need to convert (x,y,z) to slice image
                // [fetalRecontruction] coordinates because slices can have
                // [fetalRecontruction] transformations included in them (they are
                // [fetalRecontruction] nifti)  and those are not reflected in
                // [fetalRecontruction] PSF. In slice image coordinates we are
                // [fetalRecontruction] sure that z is through-plane

                // [fetalRecontruction] adjust according to voxel size
                x /= dx;
                y /= dy;
                z /= dz;
                // [fetalRecontruction] center over current voxel
                x += i;
                y += j;

                // [fetalRecontruction] convert from slice image coordinates to 
                // [fetalRecontruction] world coordinates
                slice.ImageToWorld(x, y, z);

                // [fetalRecontruction] Transform to space of reconstructed volume
                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                // [fetalRecontruction] Change to image coordinates
                reconstructor->_reconstructed.WorldToImage(x, y, z);

                // [fetalRecontruction] determine coefficients of volume 
                // [fetalRecontruction] voxels for position x,y,z
                // [fetalRecontruction] using linear interpolation

                // [fetalRecontruction] Find the 8 closest volume voxels

                // [fetalRecontruction] lowest corner of the cube
                nx = (int)floor(x);
                ny = (int)floor(y);
                nz = (int)floor(z);

                // [fetalRecontruction] not all neighbours might be in ROI, 
                // [fetalRecontruction] thus we need to
                // [fetalRecontruction] normalize
                // [fetalRecontruction] (l,m,n) are image coordinates of 8 
                // [fetalRecontruction] neighbours in volume space
                // [fetalRecontruction] for each we check whether it is in volume
                sum = 0;
                // [fetalRecontruction] to find wether the current slice 
                // [fetalRecontruction] voxel has overlap with ROI
                bool inside = false;
                for (l = nx; l <= nx + 1; l++)
                  if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                    for (m = ny; m <= ny + 1; m++)
                      if ((m >= 0) &&
                          (m < reconstructor->_reconstructed.GetY()))
                        for (n = nz; n <= nz + 1; n++)
                          if ((n >= 0) &&
                              (n < reconstructor->_reconstructed.GetZ())) {
                            weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) *
                                     (1 - fabs(n - z));
                            sum += weight;
                            if (reconstructor->_mask(l, m, n) == 1) {
                              inside = true;
                              slice_inside = 1;
			    }
                          }

		
                // [fetalRecontruction] if there were no voxels do nothing
                if ((sum <= 0) || (!inside))
                  continue;
                // [fetalRecontruction] now calculate the transformed PSF
                for (l = nx; l <= nx + 1; l++)
                  if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                    for (m = ny; m <= ny + 1; m++)
                      if ((m >= 0) &&
                          (m < reconstructor->_reconstructed.GetY()))
                        for (n = nz; n <= nz + 1; n++)
                          if ((n >= 0) &&
                              (n < reconstructor->_reconstructed.GetZ())) {
                            weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) *
                                     (1 - fabs(n - z));

                            // [fetalRecontruction] image coordinates in tPSF
                            // [fetalRecontruction] (centre,centre,centre) in 
                            // [fetalRecontruction] tPSF is aligned with
                            // [fetalRecontruction] (tx,ty,tz)
                            int aa, bb, cc;
                            aa = l - tx + centre;
                            bb = m - ty + centre;
                            cc = n - tz + centre;

                            // [fetalRecontruction] resulting value
                            double value = PSF(ii, jj, kk) * weight / sum;

                            // [fetalRecontruction] Check that we are in tPSF
                            if ((aa < 0) || (aa >= dim) || (bb < 0) ||
                                (bb >= dim) || (cc < 0) || (cc >= dim)) {
                              FORPRINTF("ERROR in runCoeffInit\n");
                            } else
                              // [fetalRecontruction] update transformed PSF
                              tPSF(aa, bb, cc) += value;
			  }
              } // [fetalRecontruction] end of the loop for PSF points

          // [fetalRecontruction] store tPSF values
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
	} // [fetalRecontruction] end of loop for slice voxels

    reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
    reconstructor->_slice_inside_cpu[inputIndex] = slice_inside;
  } // [fetalRecontruction] end of loop through the slices
      
  /*reconstructor->printvolcoeffs();
  FORPRINTF("sum bool = %d\n", sumBool(reconstructor->_slice_inside_cpu));
  FORPRINTF("sum recon = %lf\n", reconstructor->SumRecon());*/
}

class ParallelCoeffInit {
public:
  irtkReconstructionEbb *reconstructor;
  int nt;

  ParallelCoeffInit(irtkReconstructionEbb *_reconstructor, int _nt)
      : reconstructor(_reconstructor), nt(_nt) {}

  void operator()(const blocked_range<int> &r) const {
    runCoeffInit(reconstructor, r.begin(), r.end());
  }
  // [SMAFJAS] ALL THE NODES NEED TO RUN THIS FOR ALL THE SLICES 
  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<int>(0, reconstructor->_slices.size()), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::CoeffInit(int iter) {
  _volcoeffs.clear();
  _volcoeffs.resize(_slices.size());

  _slice_inside_cpu.clear();
  _slice_inside_cpu.resize(_slices.size());
  
#ifndef __EBBRT_BM__
  if (iter == 0) {
      int diff = _slices.size();
      int factor = (int)ceil(diff / (float)(numNodes+1));
      _start = 0 * factor;
      _end = 0 * factor + factor;
      _end = (_end > diff) ? diff : _end;
  
      //FORPRINTF("_start = %d, _end = %d\n", _start, _end);

      for (int i = 0; i < (int)nids.size(); i++) 
      {
	  int start = (i+1) * factor;
	  int end = (i+1) * factor + factor;
	  end = (end > diff) ? diff : end;

	  auto buf = MakeUniqueIOBuf((5 * sizeof(int)) + (4 * sizeof(double)));
	  auto dp = buf->GetMutDataPointer();
	  dp.Get<int>() = 0; // [SMAFJAS] THIS MEANS CALL COEFF INIT IN THE BACKEND
	  dp.Get<int>() = start;
	  dp.Get<int>() = end;
      
	  dp.Get<double>() = _delta;
	  dp.Get<double>() = _lambda;
	  dp.Get<double>() = _alpha;
	  dp.Get<double>() = _quality_factor;
      
	  dp.Get<int>() = _stack_factor.size();
	  dp.Get<int>() = _stack_index.size();

	  auto sf = std::make_unique<StaticIOBuf>(
	      reinterpret_cast<const uint8_t *>(_stack_factor.data()),
	      (size_t)(_stack_factor.size() * sizeof(float)));
      
	  auto si = std::make_unique<StaticIOBuf>(
	      reinterpret_cast<const uint8_t *>(_stack_index.data()),
	      (size_t)(_stack_index.size() * sizeof(int)));
      
      // [SMAFJAS] Prepending IOBUFS 
	  buf->PrependChain(std::move(SerializeSlices()));
	  buf->PrependChain(std::move(SerializeReconstructed()));
	  buf->PrependChain(std::move(SerializeMask()));
	  buf->PrependChain(std::move(SerializeTransformations()));
	  buf->PrependChain(std::move(sf));
	  buf->PrependChain(std::move(si));
      
      
	  //FORPRINTF("CoeffInit : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	  bytesTotal += buf->ComputeChainDataLength();
	  SendMessage(nids[i], std::move(buf));
      }
  }
  else
  {
      for (int i = 0; i < (int)nids.size(); i++) 
      {
	  auto buf = MakeUniqueIOBuf((1 * sizeof(int)) + (4 * sizeof(double)));
	  auto dp = buf->GetMutDataPointer();
	  dp.Get<int>() = 14;
	  dp.Get<double>() = _delta;
	  dp.Get<double>() = _lambda;
	  dp.Get<double>() = _alpha;
	  dp.Get<double>() = _quality_factor;

	  buf->PrependChain(std::move(SerializeTransformations()));

	  //FORPRINTF("CoeffInit : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	  bytesTotal += buf->ComputeChainDataLength();
	  SendMessage(nids[i], std::move(buf));
      }
  }
  
  ParallelCoeffInit coeffinit(this, _numThreads);
  coeffinit();

  //printvolcoeffs();
  //FORPRINTF("sum bool = %d\n", sumInt(_slice_inside_cpu));
  //FORPRINTF("sum recon = %lf\n", SumRecon());
  
#else
  // [SMAFJAS] BACKEND
  size_t ncpus = ebbrt::Cpu::Count();
  static ebbrt::SpinBarrier bar(ncpus);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t theCpu = ebbrt::Cpu::GetMine();
  int diff = _slices.size();
      
  for (size_t i = 0; i < ncpus; i++) {
    if (_debug)
      FORPRINTF("[BM] CoeffInit Running cpu %d of %d\n", (int) i, (int)ncpus);
      // spawn jobs on each core using SpawnRemote
      ebbrt::event_manager->SpawnRemote(
          [this, theCpu, ncpus, &count, &context, i, diff]() {
	      // get my cpu id
	      size_t mycpu = ebbrt::Cpu::GetMine();
	      int starte, ende, factor;
	      factor = (int)ceil(diff / (float)ncpus);
	      starte = i * factor;
	      ende = i * factor + factor;
	      ende = (ende > diff) ? diff : ende;

	      //FORPRINTF("%d: sum =  %lf\n", i, sumOneImage(this->_mask));
	      runCoeffInit(this, starte, ende);
	      count++;
              
          if (_debug)
            FORPRINTF("[BM] CoeffInit CPU %d barrier wait\n", mycpu);
	      bar.Wait();
          if (_debug) {
	        FORPRINTF("[BM] CoeffInit CPU %d barrier OK\n", mycpu);
          // [SMAFJAS] REVIEW THIS WHILE MAYBE HERE IS THE PLACE WHERE IT STUCKS
	        FORPRINTF("[BM] CoeffInit CPU %d inside while\n", mycpu);
          }
	      while (count < (size_t)ncpus)
		  ;
          if (_debug)
            FORPRINTF("[BM] CoeffInit CPU %d outside while\n", mycpu);
	      if (mycpu == theCpu) {
		  ebbrt::event_manager->ActivateContext(std::move(context));
	      }
          },
          indexToCPU(
              i)); // if i don't add indexToCPU, one of the cores never run ? ?
  }
  // [SMAFJAS] BY SAVING THE CONTEXT WE ARE PAUSING THE EVENT AND THEN LATER IT WILL CONTINUE
  ebbrt::event_manager->SaveContext(context);

  //printvolcoeffs();
  //FORPRINTF("sum bool = %d\n", sumInt(_slice_inside_cpu));
  //FORPRINTF("sum recon = %lf\n", SumRecon());
  
#endif
  // [SMAFJAS] THIS SHOULD PROB ONLY RUN IN THE BANCKENDS
  // prepare image for volume weights, will be needed for Gaussian
  // Reconstruction
  _volume_weights.Initialize(_reconstructed.GetImageAttributes());
  _volume_weights = 0;

  int i, j, n, k, inputIndex;
  POINT3D p;
  for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
    for (i = 0; i < _slices[inputIndex].GetX(); i++)
      for (j = 0; j < _slices[inputIndex].GetY(); j++) {
        n = _volcoeffs[inputIndex][i][j].size();
        for (k = 0; k < n; k++) {
          p = _volcoeffs[inputIndex][i][j][k];
          _volume_weights(p.x, p.y, p.z) += p.value;
        }
      }
  }

  // find average volume weight to modify alpha parameters accordingly
  irtkRealPixel *ptr = _volume_weights.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  double sum = 0;
  int num = 0;
  for (int i = 0; i < _volume_weights.GetNumberOfVoxels(); i++) {
    if (*pm == 1) {
      sum += *ptr;
      num++;
    }
    ptr++;
    pm++;
  }

  _average_volume_weight = sum / num;

  //FORPRINTF("\n********* COEFFINIT ***********\n\t_average_volume_weight = %lf\n\t_volume_weights = %f\n\tchecksum _reconstructed = %f\n*********************\n", _average_volume_weight, sumOneImage(_volume_weights), SumRecon());
  
  
#ifndef __EBBRT_BM__
  // [SMAFJAS] BLOCK UNTILS IT GETS ALL THE DATA BACK FROM THE BACKEND
  if (_debug)
    FORPRINTF("[H] CoeffInit: Blocking ... \n");
  testFuture = ebbrt::Promise<int>(); // [SMAFJAS] int 1: SUCCESS
  auto tf = testFuture.GetFuture();
  tf.Block();
  if (_debug)
    FORPRINTF("[H] CoeffInit: returned from future\n");
#endif

} // end of CoeffInit()

// [RETURN HERE]

void irtkReconstructionEbb::GaussianReconstruction() {

  unsigned int inputIndex;
  int i, j, k, n;
  irtkRealImage slice;
  double scale;
  POINT3D p;
  int slice_vox_num;

#ifndef __EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf(1 * sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 1;

      //FORPRINTF("GaussianReconstruction : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
  _voxel_num.resize(_slices.size());
#else
  _voxel_num.resize(_end-_start);
#endif
  
  // [fetalRecontruction] clear _reconstructed image
  _reconstructed = 0;

  for (inputIndex = _start; inputIndex < _end; ++inputIndex) {
    // [fetalRecontruction] copy the current slice
    slice = _slices[inputIndex];
    // [fetalRecontruction] alias the current bias image
    irtkRealImage &b = _bias[inputIndex];
    // [fetalRecontruction] read current scale factor
    scale = _scale_cpu[inputIndex];

    slice_vox_num = 0;

    // [fetalRecontruction] Distribute slice intensities to the volume
    for (i = 0; i < slice.GetX(); i++)
    {
	for (j = 0; j < slice.GetY(); j++)
	{
	    if (slice(i, j, 0) != -1) 
	    {
		// [fetalRecontruction] biascorrect and scale the slice
		slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
	    
		// [fetalRecontruction] number of volume voxels with non-zero coefficients
		// [fetalRecontruction] for current slice voxel
		n = _volcoeffs[inputIndex][i][j].size();
	    
		// [fetalRecontruction] calculate num of vox in a slice that have 
        // [fetalRecontruction] overlap with roi
		if (n > 0)
		    slice_vox_num++;
	    
		// [fetalRecontruction] add contribution of current slice voxel to all 
        // [fetalRecontruction] voxel volumes to which it contributes
		for (k = 0; k < n; k++) {
		    p = _volcoeffs[inputIndex][i][j][k];
		    _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
		}
	    }
	}
    }
    _voxel_num[inputIndex-_start] = slice_vox_num;
  } // [fetalRecontruction] end of loop for a slice inputIndex
  

    //FORPRINTF("\n*********** GaussianReconstruction 1 ***********\n_reconstructed = %lf, _volume_weights = %lf\n***************\n", SumRecon(), sumOneImage(_volume_weights));
  
  // [fetalRecontruction] normalize the volume by proportion of contributing 
  // [fetalRecontruction] slice voxels for each volume voxe
  _reconstructed /= _volume_weights;

  //FORPRINTF("\n*********** GaussianReconstruction 2 ***********\n_reconstructed = %lf, _volume_weights = %lf\n***************\n", SumRecon(), sumOneImage(_volume_weights));

#ifdef __EBBRT_BM__
  // sending
  auto retbuf = MakeUniqueIOBuf(3 * sizeof(int));
  auto retdp = retbuf->GetMutDataPointer();
  
  retdp.Get<int>() = 1;
  retdp.Get<int>() = _start;
  retdp.Get<int>() = _end;
  // [SMAFJAS] SEND BACK DATA TO THE FRONTEND
  auto vnum = std::make_unique<StaticIOBuf>(
      reinterpret_cast<const uint8_t *>(_voxel_num.data()),
      (size_t)((_end-_start) * sizeof(int)));
  
  retbuf->PrependChain(std::move(vnum));
  retbuf->PrependChain(std::move(serializeSlices(_reconstructed)));// [SMAFJAS] ACTUAL RECONSTRUCTED DATA
  
  //FORPRINTF("GaussianReconstruction : Sending %d bytes\n", (int)retbuf->ComputeChainDataLength());
  SendMessage(nids[0], std::move(retbuf));
#endif
  
//block here before continuing
#ifndef __EBBRT_BM__
  // [SMAFJAS] WAIT FOR ALL DATA
  if (_debug)  
    FORPRINTF("[H] GaussianReconstruction: Blocking\n");   
  gaussianreconFuture = ebbrt::Promise<int>();
  auto tf = gaussianreconFuture.GetFuture();
  tf.Block();
  if (_debug)  
    FORPRINTF("[H] GaussianReconstruction: returned from future\n");   
#endif
   
  // [fetalRecontruction] now find slices with small overlap with ROI 
  // [fetalRecontruction] and exclude them.
  vector<int> voxel_num_tmp;
  for (i = 0; i < _voxel_num.size(); i++)
  {
      voxel_num_tmp.push_back(_voxel_num[i]);
  }
  
  // [fetalRecontruction] find median
  sort(voxel_num_tmp.begin(), voxel_num_tmp.end());
  int median = voxel_num_tmp[round(voxel_num_tmp.size() * 0.5)];

  // [fetalRecontruction] remember slices with small overlap with ROI
  _small_slices.clear();
  for (i = 0; i < _voxel_num.size(); i++)
  {
      if (_voxel_num[i] < 0.1 * median) { _small_slices.push_back(i); }
  }
  //FORPRINTF("Gaussian Recconstruction finished\n");
}

void irtkReconstructionEbb::InitializeEM() {
  _weights.clear();
  _bias.clear();
  _scale_cpu.clear();
  _slice_weight_cpu.clear();
  slice_potential.clear();

  for (int i = 0; i < _slices.size(); i++) {
    // [fetalRecontruction] Create images for voxel weights and bias fields
    _weights.push_back(_slices[i]);
    _bias.push_back(_slices[i]);

    // [fetalRecontruction] Create and initialize scales
    _scale_cpu.push_back(1);

    // [fetalRecontruction] Create and initialize slice weights
    _slice_weight_cpu.push_back(1);

    slice_potential.push_back(0);
  }

// [fetalRecontruction] TODO CUDA
// [fetalRecontruction] Find the range of intensities
//#ifndef __EBBRT_BM__
  _max_intensity = voxel_limits<irtkRealPixel>::min();
  _min_intensity = voxel_limits<irtkRealPixel>::max();
  for (unsigned int i = 0; i < _slices.size(); i++) {
    // [fetalRecontruction] to update minimum we need to exclude padding value
    irtkRealPixel *ptr = _slices[i].GetPointerToVoxels();
    for (int ind = 0; ind < _slices[i].GetNumberOfVoxels(); ind++) {
      if (*ptr > 0) {
        if (*ptr > _max_intensity)
          _max_intensity = *ptr;
        if (*ptr < _min_intensity)
          _min_intensity = *ptr;
      }
      ptr++;
    }
  }

  //FORPRINTF("_max_intensity = %lf, _mmin_intensity = %lf\n", _max_intensity, _min_intensity);
//#endif
}

// _slices never gets updated, so InitializeEMValues
// always produces the same result, can run this
// function independently on each node
void irtkReconstructionEbb::InitializeEMValues() {
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
    _slice_weight_cpu[i] = 1;

    // [fetalRecontruction] Initialize scaling factors for intensity matching
    _scale_cpu[i] = 1;
  }

  //FORPRINTF("\n***** InitializeEMValues **********\n");
  //FORPRINTF(" _weights = %f ", sumImage(_weights));
  //FORPRINTF("_bias = %f ", sumImage(_bias));
  //FORPRINTF(" _slices = %f ", sumImage(_slices));
  //FORPRINTF("_slice_weight_cpu = %lf ", sumVec(_slice_weight_cpu));
  //FORPRINTF("_scale_cpu = %lf *** \n", sumVec(_scale_cpu));
  //FORPRINTF("*****************************\n\n");
}

void irtkReconstructionEbb::InitializeRobustStatistics() {
  // [fetalRecontruction] Initialize parameter of EM robust statistics
  int i, j;
  irtkRealImage slice, sim;
  _tsigma = 0.0;
  _tnum = 0;
  
#ifndef __EBBRT_BM__
  for (i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf(1 * sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 3;
      //FORPRINTF("InitializeRobustStatistics : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
#endif

  for (unsigned int inputIndex = _start; inputIndex < _end; inputIndex++) {
    slice = _slices[inputIndex];

    // [fetalRecontruction] Voxel-wise sigma will be set to stdev of volumetric 
    // [fetalRecontruction] errors
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] calculate stev of the errors
          if ((_simulated_inside[inputIndex](i, j, 0) == 1) &&
              (_simulated_weights[inputIndex](i, j, 0) > 0.99)) {
            slice(i, j, 0) -= _simulated_slices[inputIndex](i, j, 0);
            _tsigma += slice(i, j, 0) * slice(i, j, 0);
            _tnum++;
          }
        }

    // [fetalRecontruction] if slice does not have an overlap with ROI, 
    // [fetalRecontruction] set its weight to zero
    if (!_slice_inside_cpu[inputIndex])
      _slice_weight_cpu[inputIndex] = 0;
  }
  
#ifndef __EBBRT_BM__
    if (_debug)  
      FORPRINTF("[H] RobustStatistics: Bloacking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf = testFuture.GetFuture();
    tf.Block();
    if (_debug)  
      FORPRINTF("[H] RobustStatistics: Returning from Future \n");

  // [fetalRecontruction] Force exclusion of slices predefined by user 
  // [EbbRT] not needed - Han
  // for (unsigned int i = 0; i < _force_excluded.size(); i++)
  //  _slice_weight_cpu[_force_excluded[i]] = 0;

  // [fetalRecontruction] initialize sigma for voxelwise robust statistics
  _sigma_cpu = _tsigma / _tnum;
  // [fetalRecontruction] initialize sigma for slice-wise robust statistics
  _sigma_s_cpu = 0.025;
  // [fetalRecontruction] initialize mixing proportion for inlier class in 
  // [fetalRecontruction] voxel-wise robust statistics
  _mix_cpu = 0.9;
  // [fetalRecontruction] initialize mixing proportion for outlier class in 
  // [fetalRecontruction] slice-wise robust statistics
  _mix_s_cpu = 0.9;
  // [fetalRecontruction] Initialise value for uniform distribution according to 
  // [fetalRecontruction] the range of intensities
  _m_cpu = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);
  
//FORPRINTF("\nInitializeRobustStatistics: \n\t_sigma_cpu=%lf\n\tsigma_s_cpu=%lf\n\t_mix_cpu=%lf\n\t_mix_s_cpu=%lf\n\t_m_cpu=%lf\n\t_tsigma=%lf, _tnum=%d\n",  _sigma_cpu, _sigma_s_cpu, _mix_cpu, _mix_s_cpu, _m_cpu, _tsigma, _tnum);

#else
  auto retbuf = MakeUniqueIOBuf((2 * sizeof(int)) + (1 * sizeof(double)));
  auto retdp = retbuf->GetMutDataPointer();
  
  retdp.Get<int>() = 3;
  retdp.Get<int>() = _tnum;
  retdp.Get<double>() = _tsigma;

  //FORPRINTF("InitializeRobustStatistics : Sending %d bytes\n", (int)retbuf->ComputeChainDataLength());
  SendMessage(nids[0], std::move(retbuf));
#endif
}

void runEStep(irtkReconstructionEbb *reconstructor, int start, int end, double &sum, double &den, double &sum2, double &den2, double &maxs, double &mins) {
  for (int inputIndex = start; inputIndex < end; inputIndex++) {
    // [fetalRecontruction] read the current slice
    irtkRealImage slice = reconstructor->_slices[inputIndex];

    // [fetalRecontruction] alias the current bias image
    irtkRealImage &b = reconstructor->_bias[inputIndex];

    // [fetalRecontruction] identify scale factor
    double scale = reconstructor->_scale_cpu[inputIndex];

    double num = 0;
    // [fetalRecontruction] Calculate error, voxel weights, and slice potential
    for (int i = 0; i < slice.GetX(); i++) {
      for (int j = 0; j < slice.GetY(); j++) {
        if (slice(i, j, 0) != -1) {
          // [fetalRecontruction] bias correct and scale the slice
          slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

          // [fetalRecontruction] number of volumetric voxels to which
          // [fetalRecontruction] current slice voxel contributes
          int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

          // [fetalRecontruction] if n == 0, slice voxel has no overlap with 
          // [fetalRecontruction] volumetric ROI, do not process it

          if ((n > 0) &&
              (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0)) {
            slice(i, j, 0) -=
                reconstructor->_simulated_slices[inputIndex](i, j, 0);

            // [fetalRecontruction] calculate norm and voxel-wise weights
            // [fetalRecontruction] Gaussian distribution for inliers (likelihood)
            double g =
                reconstructor->G(slice(i, j, 0), reconstructor->_sigma_cpu);
            // [fetalRecontruction] Uniform distribution for outliers (likelihood)
            double m = reconstructor->M(reconstructor->_m_cpu);

            // [fetalRecontruction] voxel_wise posterior
            double weight = g * reconstructor->_mix_cpu /
                            (g * reconstructor->_mix_cpu +
                             m * (1 - reconstructor->_mix_cpu));
            reconstructor->_weights[inputIndex](i, j, 0) = weight;
            // [fetalRecontruction] calculate slice potentials
            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
              reconstructor->slice_potential[inputIndex] +=
                  (1.0 - weight) * (1.0 - weight);
              num++;
            }
          } else {
            reconstructor->_weights[inputIndex](i, j, 0) = 0;
          }
        }
      }
    }

    // [fetalRecontruction] evaluate slice potential
    if (num > 0) {
      reconstructor->slice_potential[inputIndex] =
          sqrt(reconstructor->slice_potential[inputIndex] / num);
    } 
    else
    {
      // [fetalRecontruction] slice has no unpadded voxels
      reconstructor->slice_potential[inputIndex] = -1; 
    }

    for(int i = 0; i < reconstructor->_small_slices.size(); i++)
    {
	reconstructor->slice_potential[reconstructor->_small_slices[i]] = -1;
    }

    if ((reconstructor->_scale_cpu[inputIndex] < 0.2) || (reconstructor->_scale_cpu[inputIndex] > 5)) 
    {
	reconstructor->slice_potential[inputIndex] = -1;
    }

    if (reconstructor->slice_potential[inputIndex] >= 0)
    {
	// calculate means
	sum += reconstructor->slice_potential[inputIndex] * reconstructor->_slice_weight_cpu[inputIndex];
	den += reconstructor->_slice_weight_cpu[inputIndex];
	sum2 += reconstructor->slice_potential[inputIndex] * (1 - reconstructor->_slice_weight_cpu[inputIndex]);
	den2 += (1 - reconstructor->_slice_weight_cpu[inputIndex]);
	
	// calculate min and max of potentials in case means need to be initalized
	if (reconstructor->slice_potential[inputIndex] > maxs)
	    maxs = reconstructor->slice_potential[inputIndex];
	if (reconstructor->slice_potential[inputIndex] < mins)
	    mins = reconstructor->slice_potential[inputIndex];
    }
  }
}

class ParallelEStep {
    irtkReconstructionEbb *reconstructor;
    vector<double> &slice_potential;
    int nt;
    int start;
    int end;
    
public:
    double sum;
    double den;
    double sum2;
    double den2;
    double maxs;
    double mins;
    
    void operator()(const blocked_range<int> &r) {
	runEStep(reconstructor, r.begin(), r.end(), sum, den, sum2, den2, maxs, mins);
    }

    ParallelEStep(ParallelEStep &x, split)
	: reconstructor(x.reconstructor), slice_potential(x.slice_potential)
	{
	    nt = x.nt;
	    start = x.start;
	    end = x.end;
	    
	    sum = 0.0;
	    den = 0.0;
	    sum2 = 0.0;
	    den2 = 0.0;
	    maxs = 0.0;
	    mins = 0.0;
	}
    
    ParallelEStep(irtkReconstructionEbb *reconstructor,
		  vector<double> &slice_potential,
		  int _nt, int _start, int _end)
	: reconstructor(reconstructor), slice_potential(slice_potential)
	{
	    nt = _nt;
	    start = _start;
	    end = _end;
	    sum = den = sum2 = den2 = maxs = 0;
	    mins = voxel_limits<irtkRealPixel>::max();
	}

  void join(const ParallelEStep &y) {
      sum += y.sum;
      den += y.den;
      sum2 += y.sum2;
      den2 += y.den2;
      if(y.maxs > maxs) maxs = y.maxs;
      if(y.mins < mins) mins = y.mins;
  }

  void operator()() {
    task_scheduler_init init(nt);
    parallel_reduce(blocked_range<int>(0, (end-start)), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::EStep() {
    size_t inputIndex;
    irtkRealImage slice, w, b, sim;
    int num = 0;
    std::fill(slice_potential.begin(), slice_potential.end(), 0);
	
#ifndef __EBBRT_BM__
    for (int i = 0; i < (int)nids.size(); i++) 
    {
	auto buf = MakeUniqueIOBuf((2 * sizeof(int)) + (5 * sizeof(double)));
	auto dp = buf->GetMutDataPointer();
	dp.Get<int>() = 4;
	dp.Get<double>() = _sigma_cpu;
	dp.Get<double>() = _sigma_s_cpu;
	dp.Get<double>() = _mix_cpu;
	dp.Get<double>() = _mix_s_cpu;
	dp.Get<double>() = _m_cpu;
	
	dp.Get<int>() = _small_slices.size();
	
	auto vnum = std::make_unique<StaticIOBuf>(
	    reinterpret_cast<const uint8_t *>(_small_slices.data()),
	    (size_t)(_small_slices.size() * sizeof(int)));

	buf->PrependChain(std::move(vnum));
	
	//FORPRINTF("EStep : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	bytesTotal += buf->ComputeChainDataLength();
	SendMessage(nids[i], std::move(buf));
    }
    ParallelEStep parallelEStep(this, slice_potential, _numThreads, _start, _end);
    parallelEStep();

    _tsum = parallelEStep.sum;
    _tden = parallelEStep.den;
    _tsum2 = parallelEStep.sum2;
    _tden2 = parallelEStep.den2;
    _tmaxs = parallelEStep.maxs;
    _tmins = parallelEStep.mins;

    if (_debug)  
      FORPRINTF("[H] EStep: Bloacking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf = testFuture.GetFuture();
    tf.Block();
    if (_debug)  
      FORPRINTF("[H] Estep: Returning from future \n");
    
#else

    _tsum = _tden = _tsum2 = _tden2 = _tmaxs = 0;
    _tmins = 1;

    size_t ncpus = ebbrt::Cpu::Count();
    if (ncpus > 1) {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = (_end-_start);
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = (i * factor) + _start;
		    ende = (i * factor + factor) + _start;
		    ende = (ende > _end) ? _end : ende;

		    double sum, den, sum2, den2, maxs, mins;
		    sum = den = sum2 = den2 = maxs = 0;
		    mins = 1;
		    
		    runEStep(this, starte, ende, sum, den, sum2, den2, maxs, mins);
		    // braces for scoping
		    {
			// lock_guard auto unlocks end of scope, mutex doesn't work yet
			std::lock_guard<ebbrt::SpinLock> l(spinlock);
			_tsum += sum;
			_tden += den;
			_tsum2 += sum2;
			_tden2 += den2;
			if(maxs > _tmaxs) _tmaxs = maxs;
			if(mins < _tmins) _tmins = mins;
		    }

		    count++;
            if (_debug)  
		      FORPRINTF("[BM] EStep CPU %d barrier wait\n", mycpu);
		    bar.Wait();
            if (_debug) { 
		      FORPRINTF("[BM] EStep CPU %d barrier done\n", mycpu);
		      FORPRINTF("[BM] EStep CPU %d while wait\n", mycpu);
            }
		    while (count < (size_t)ncpus)
			;
            if (_debug)  
		      FORPRINTF("[BM] EStep CPU %d while done\n", mycpu);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		indexToCPU(
		    i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    
    //FORPRINTF("runEStep\n");
    //runEStep(this, _start, _end, _tsum, _tden, _tsum2, _tden2, _tmaxs, _tmins);
    
    auto retbuf = MakeUniqueIOBuf((6 * sizeof(double)) + (1 * sizeof(int)));
    auto retdp = retbuf->GetMutDataPointer();
  
    retdp.Get<int>() = 4;
    retdp.Get<double>() = _tsum;
    retdp.Get<double>() = _tden; 
    retdp.Get<double>() = _tsum2; 
    retdp.Get<double>() = _tden2; 
    retdp.Get<double>() = _tmaxs; 
    retdp.Get<double>() = _tmins; 
				  
    //FORPRINTF("EStep : Sending %d bytes\n", (int)retbuf->ComputeChainDataLength());
    SendMessage(nids[0], std::move(retbuf));

    return;
#endif
    //FORPRINTF("%lf %lf %lf %lf %lf %lf\n", _tsum, _tden, _tsum2, _tden2, _tmaxs, _tmins);
  
    if (_tden > 0)
	_mean_s_cpu = _tsum / _tden;
    else
	_mean_s_cpu = _tmins;
    
    if (_tden2 > 0)
	_mean_s2_cpu = _tsum2 / _tden2;
    else
	_mean_s2_cpu = (_tmaxs + _mean_s_cpu) / 2;

    //FORPRINTF("%lf %lf\n", _mean_s_cpu, _mean_s2_cpu);
    
#ifndef __EBBRT_BM__
    for (int i = 0; i < (int)nids.size(); i++) 
    {
	auto buf = MakeUniqueIOBuf((1 * sizeof(int)) + (2 * sizeof(double)));
	auto dp = buf->GetMutDataPointer();
	dp.Get<int>() = 5;
	dp.Get<double>() = _mean_s_cpu;
	dp.Get<double>() = _mean_s2_cpu;

	//FORPRINTF("EStep : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	bytesTotal += buf->ComputeChainDataLength();
	SendMessage(nids[i], std::move(buf));
    }
#endif    
    
    // [fetalRecontruction] Calculate the variances of the potentials
    _ttsum = 0;
    _ttden = 0;
    _ttsum2 = 0;
    _ttden2 = 0;

    for (inputIndex = _start; inputIndex < _end; inputIndex++)
    {
	if (slice_potential[inputIndex] >= 0) 
	{
	    _ttsum += (slice_potential[inputIndex] - _mean_s_cpu) *
		(slice_potential[inputIndex] - _mean_s_cpu) *
		_slice_weight_cpu[inputIndex];
	  
	    _ttden += _slice_weight_cpu[inputIndex];
	  
	    _ttsum2 += (slice_potential[inputIndex] - _mean_s2_cpu) *
		(slice_potential[inputIndex] - _mean_s2_cpu) *
		(1 - _slice_weight_cpu[inputIndex]);
	  
	    _ttden2 += (1 - _slice_weight_cpu[inputIndex]);
	}
    }
    if (_debug)  
      FORPRINTF("[H] EStep: Blocking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf2 = testFuture.GetFuture();
    tf2.Block();
    if (_debug)  
      FORPRINTF("[H] Estep: Returning from future \n");
  
    //FORPRINTF("%lf %lf %lf %lf\n", _ttsum, _ttden, _ttsum2, _ttden2);
    //return;
  
  if ((_ttsum > 0) && (_ttden > 0)) {
    _sigma_s_cpu = _ttsum / _ttden;
    // [fetalRecontruction] do not allow too small sigma
    if (_sigma_s_cpu < _step * _step / 6.28)
      _sigma_s_cpu = _step * _step / 6.28;
  } else {
    _sigma_s_cpu = 0.025;
  }

  if ((_ttsum2 > 0) && (_ttden2 > 0)) {
    _sigma_s2_cpu = _ttsum2 / _ttden2;
    // [fetalRecontruction] do not allow too small sigma
    if (_sigma_s2_cpu < _step * _step / 6.28)
      _sigma_s2_cpu = _step * _step / 6.28;
  } else {
    _sigma_s2_cpu =
        (_mean_s2_cpu - _mean_s_cpu) * (_mean_s2_cpu - _mean_s_cpu) / 4;
    // [fetalRecontruction] do not allow too small sigma
    if (_sigma_s2_cpu < _step * _step / 6.28)
      _sigma_s2_cpu = _step * _step / 6.28;
  }


#ifndef __EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf((1 * sizeof(int)) + (2 * sizeof(double)));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 6;
      dp.Get<double>() = _sigma_s_cpu;
      dp.Get<double>() = _sigma_s2_cpu;
      
      //FORPRINTF("EStep : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
#endif

  // [fetalRecontruction] Calculate slice weights
  double gs1, gs2;
  //double sum = 0;
  //num = 0;
  _ttsum = 0;
  _ttnum = 0;
  
  for (inputIndex = _start; inputIndex < _end; inputIndex++)
{
    // [fetalRecontruction] Slice does not have any voxels in volumetric ROI
    if (slice_potential[inputIndex] == -1) {
      _slice_weight_cpu[inputIndex] = 0;
      continue;
    }

    // [fetalRecontruction] All slices are outliers or the means are not valid
    if ((_ttden <= 0) || (_mean_s2_cpu <= _mean_s_cpu)) {
      _slice_weight_cpu[inputIndex] = 1;
      continue;
    }

    // [fetalRecontruction] likelihood for inliers
    if (slice_potential[inputIndex] < _mean_s2_cpu)
      gs1 = G(slice_potential[inputIndex] - _mean_s_cpu, _sigma_s_cpu);
    else
      gs1 = 0;

    // [fetalRecontruction] likelihood for outliers
    if (slice_potential[inputIndex] > _mean_s_cpu)
      gs2 = G(slice_potential[inputIndex] - _mean_s2_cpu, _sigma_s2_cpu);
    else
      gs2 = 0;

    // [fetalRecontruction] calculate slice weight
    double likelihood = gs1 * _mix_s_cpu + gs2 * (1 - _mix_s_cpu);
    if (likelihood > 0)
      _slice_weight_cpu[inputIndex] = gs1 * _mix_s_cpu / likelihood;
    else {
      if (slice_potential[inputIndex] <= _mean_s_cpu)
        _slice_weight_cpu[inputIndex] = 1;
      if (slice_potential[inputIndex] >= _mean_s2_cpu)
        _slice_weight_cpu[inputIndex] = 0;
      // [fetalRecontruction] should not happen
      if ((slice_potential[inputIndex] < _mean_s2_cpu) &&
          (slice_potential[inputIndex] > _mean_s_cpu)) 
        _slice_weight_cpu[inputIndex] = 1;
    }

    if (slice_potential[inputIndex] >= 0) {
	_ttsum += _slice_weight_cpu[inputIndex];
	_ttnum ++;
    }
  }

    if (_debug) 
      FORPRINTF("[H] EStep: Bloacking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf3 = testFuture.GetFuture();
    tf3.Block();
    if (_debug) 
      FORPRINTF("[H] Estep: Returning from future \n");
    
    if (_ttnum > 0)
	_mix_s_cpu = _ttsum / _ttnum;
    else {
	_mix_s_cpu = 0.9;
    }
    
    //FORPRINTF("EStep: %d %lf %lf\n", _ttnum, _ttsum, _mix_s_cpu);
}

void runScale(irtkReconstructionEbb *reconstructor, int start, int end) {
  for (int inputIndex = start; inputIndex != end; inputIndex++) {
    // [fetalRecontruction] alias the current slice
    irtkRealImage &slice = reconstructor->_slices[inputIndex];

    // [fetalRecontruction] alias the current weight image
    irtkRealImage &w = reconstructor->_weights[inputIndex];

    // [fetalRecontruction] alias the current bias image
    irtkRealImage &b = reconstructor->_bias[inputIndex];

    // [fetalRecontruction] initialise calculation of scale
    double scalenum = 0;
    double scaleden = 0;

    for (int i = 0; i < slice.GetX(); i++)
      for (int j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
            // [fetalRecontruction] scale - intensity matching
            double eb = exp(-b(i, j, 0));
            scalenum += w(i, j, 0) * slice(i, j, 0) * eb *
                        reconstructor->_simulated_slices[inputIndex](i, j, 0);
            scaleden += w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
          }
        }

    // [fetalRecontruction] calculate scale for this slice
    if (scaleden > 0)
      reconstructor->_scale_cpu[inputIndex] = scalenum / scaleden;
    else
      reconstructor->_scale_cpu[inputIndex] = 1;
  }
}

class ParallelScale {
    irtkReconstructionEbb *reconstructor;
    int nt;
    int start;
    int end;

public:
    ParallelScale(irtkReconstructionEbb *_reconstructor, int _nt, int _start, int _end)
	: reconstructor(_reconstructor), nt(_nt), start(_start), end(_end) {}

  void operator()(const blocked_range<int> &r) const {
    runScale(reconstructor, r.begin(), r.end());
  }

  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<int>(0, end-start), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::Scale() {
    //FORPRINTF("In Scale()\n");
#ifndef __EBBRT_BM__
    for (int i = 0; i < (int)nids.size(); i++) 
    {
       auto buf = MakeUniqueIOBuf(1 * sizeof(int));
       auto dp = buf->GetMutDataPointer();
       dp.Get<int>() = 7;
       //FORPRINTF("Scale : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
       bytesTotal += buf->ComputeChainDataLength();
       SendMessage(nids[i], std::move(buf));
    }
    ParallelScale scale(this, _numThreads, _start, _end);
    scale();
    //FORPRINTF("Scale() : _scale_cpu = %lf\n", sumPartVec(_scale_cpu, _start, _end));
    
    if (_debug) 
      FORPRINTF("[H] Scale: Blocking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf = testFuture.GetFuture();
    tf.Block();
    if (_debug) 
      FORPRINTF("[H] Scale: Returning from future \n");
#else
    
    // [EbbRT] EbbRT code, figure out what it does and if it does
    // not work delete it

    /*size_t ncpus = ebbrt::Cpu::Count();
    static ebbrt::SpinBarrier bar(ncpus);
    ebbrt::EventManager::EventContext context;
    std::atomic<size_t> count(0);
    size_t theCpu = ebbrt::Cpu::GetMine();
    int diff = (_end-_start);
    for (size_t i = 0; i < ncpus; i++) {
	// spawn jobs on each core using SpawnRemote
	ebbrt::event_manager->SpawnRemote(
	    [this, theCpu, ncpus, &count, &context, i, diff]() {
		// get my cpu id
		//FORPRINTF("%d\n",
		size_t mycpu = ebbrt::Cpu::GetMine();
		int starte, ende, factor;
		factor = (int)ceil(diff / (float)ncpus);
		starte = (i * factor) + _start;
		ende = (i * factor + factor) + _start;
		ende = (ende > _end) ? _end : ende;
		runScale(this, starte, ende);
		count++;
		bar.Wait();
		while (count < (size_t)ncpus)
		    ;
		if (mycpu == theCpu) {
		    ebbrt::event_manager->ActivateContext(std::move(context));
		}
	    },
	    indexToCPU(
		i)); // if i don't add indexToCPU, one of the cores never run ? ?
    }
    ebbrt::event_manager->SaveContext(context);\
    FORPRINTF("runScale\n");
    return;*/
    
    runScale(this, _start, _end);
    //FORPRINTF("Scale() : _scale_cpu = %lf\n", sumPartVec(_scale_cpu, _start, _end));
#endif

}

// [EbbRT] This code corresponds to the ParallelBias class of the
// original code
void irtkReconstructionEbb::Bias() {

  size_t inputIndex = 0;
  for (inputIndex = 0; inputIndex != _slices.size(); inputIndex++) {
    // [fetalReconstruction] read the current slice
    irtkRealImage slice = _slices[inputIndex];

    // [fetalReconstruction] alias the current weight image
    irtkRealImage &w = _weights[inputIndex];

    // [fetalReconstruction] alias the current bias image
    irtkRealImage b = _bias[inputIndex];

    // [fetalReconstruction] identify scale factor
    double scale = _scale_cpu[inputIndex];

    // [fetalReconstruction] prepare weight image for bias field
    irtkRealImage wb = w;

    // [fetalReconstruction] simulated slice
    irtkRealImage wresidual(slice.GetImageAttributes());
    wresidual = 0;

    for (int i = 0; i < slice.GetX(); i++)
      for (int j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          if (_simulated_weights[inputIndex](i, j, 0) > 0.99) {
            // [fetalReconstruction] bias-correct and scale current slice
            double eb = exp(-b(i, j, 0));
            slice(i, j, 0) *= (eb * scale);

            // [fetalReconstruction] calculate weight image
            wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);

            // [fetalReconstruction] calculate weighted residual image
            // [fetalReconstruction],make sure it is far from zero to avoid 
            // [fetalReconstruction] numerical instability
            if ((_simulated_slices[inputIndex](i, j, 0) > 1) &&
                (slice(i, j, 0) > 1)) {
              wresidual(i, j, 0) =
                  log(slice(i, j, 0) / _simulated_slices[inputIndex](i, j, 0)) *
                  wb(i, j, 0);
            }
          } else {
            // [fetalReconstruction] do not take into account this voxel when 
            // [fetalReconstruction] calculating bias field
            wresidual(i, j, 0) = 0;
            wb(i, j, 0) = 0;
          }
        }

    // [fetalReconstruction] calculate bias field for this slice
    irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
    // [fetalReconstruction] smooth weighted residual
    gb.SetInput(&wresidual);
    gb.SetOutput(&wresidual);
    gb.Run();

    // [fetalReconstruction] smooth weight image
    gb.SetInput(&wb);
    gb.SetOutput(&wb);
    gb.Run();

    // [fetalReconstruction] update bias field
    double sum = 0;
    double num = 0;
    for (int i = 0; i < slice.GetX(); i++)
      for (int j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          if (wb(i, j, 0) > 0)
            b(i, j, 0) += wresidual(i, j, 0) / wb(i, j, 0);
          sum += b(i, j, 0);
          num++;
        }

    // [fetalReconstruction] normalize bias field to have zero mean
    if (!_global_bias_correction) {
      double mean = 0;
      if (num > 0)
        mean = sum / num;
      for (int i = 0; i < slice.GetX(); i++)
        for (int j = 0; j < slice.GetY(); j++)
          if ((slice(i, j, 0) != -1) && (num > 0)) {
            b(i, j, 0) -= mean;
          }
    }

    _bias[inputIndex] = b;
  }
}

void irtkReconstructionEbb::AdaptiveRegularization2(vector<irtkRealImage> &_b,
                                                    vector<double> &_factor,
                                                    irtkRealImage &_original) {
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
                   _confidence_map(xx, yy, zz);
            valW += _b[i](x, y, z) * _confidence_map(xx, yy, zz);
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
                   _confidence_map(xx, yy, zz);
            valW += _b[i](xx, yy, zz) * _confidence_map(xx, yy, zz);
            sum += _b[i](xx, yy, zz);
          }
        }

        val -= sum * _original(x, y, z) * _confidence_map(x, y, z);
        valW -= sum * _confidence_map(x, y, z);
        val = _original(x, y, z) * _confidence_map(x, y, z) +
              _alpha * _lambda / (_delta * _delta) * val;
        valW = _confidence_map(x, y, z) +
               _alpha * _lambda / (_delta * _delta) * valW;

        if (valW > 0) {
          _reconstructed(x, y, z) = val / valW;
        } else
          _reconstructed(x, y, z) = 0;
      }
  }
}

void irtkReconstructionEbb::AdaptiveRegularization1(vector<irtkRealImage> &_b,
                                                    vector<double> &_factor,
                                                    irtkRealImage &_original) {
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
              (zz < dz) && (_confidence_map(x, y, z) > 0) &&
              (_confidence_map(xx, yy, zz) > 0)) {
            diff = (_original(xx, yy, zz) - _original(x, y, z)) *
                   sqrt(_factor[i]) / _delta;
            _b[i](x, y, z) = _factor[i] / sqrt(1 + diff * diff);

          } else
            _b[i](x, y, z) = 0;
        }
  }
}

void irtkReconstructionEbb::AdaptiveRegularization(int iter,
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
    PRINTF("Warning: regularization might not have smoothing effect! Ensure "
           "that alpha*lambda/delta^2 is below 0.068.\n");
  }
}

// [EbbRT] This corresponds to the Parallel Superresolution part
// of the original code
void runSuperresolution(irtkReconstructionEbb *reconstructor, int start,
                        int end, irtkRealImage &addon,
                        irtkRealImage &_confidence_map) {
  for (int inputIndex = start; inputIndex < end; ++inputIndex) {
    // [fetalReconstruction] read the current slice
    irtkRealImage slice = reconstructor->_slices[inputIndex];

    // [fetalReconstruction] read the current weight image
    irtkRealImage &w = reconstructor->_weights[inputIndex];

    // [fetalReconstruction] read the current bias image
    irtkRealImage &b = reconstructor->_bias[inputIndex];

    // [fetalReconstruction] identify scale factor
    double scale = reconstructor->_scale_cpu[inputIndex];

    // [fetalReconstruction] Update reconstructed volume using current slice
    // [fetalReconstruction] Distribute error to the volume
    POINT3D p;
    for (int i = 0; i < slice.GetX(); i++)
      for (int j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // [fetalReconstruction] bias correct and scale the slice
          slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

          if (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 0)
            slice(i, j, 0) -=
                reconstructor->_simulated_slices[inputIndex](i, j, 0);
          else
            slice(i, j, 0) = 0;

          int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
          for (int k = 0; k < n; k++) {
            p = reconstructor->_volcoeffs[inputIndex][i][j][k];
            addon(p.x, p.y, p.z) +=
                p.value * slice(i, j, 0) * w(i, j, 0) *
                reconstructor->_slice_weight_cpu[inputIndex];
            _confidence_map(p.x, p.y, p.z) +=
                p.value * w(i, j, 0) *
                reconstructor->_slice_weight_cpu[inputIndex];
          }
        }
  }
}

class ParallelSuperresolution {
    irtkReconstructionEbb *reconstructor;
    int nt;
    int start;
    int end;
public:
  irtkRealImage confidence_map;
  irtkRealImage addon;

  void operator()(const blocked_range<int> &r) {
    runSuperresolution(reconstructor, r.begin(), r.end(), addon,
                       confidence_map);
  }

  ParallelSuperresolution(ParallelSuperresolution &x, split)
      : reconstructor(x.reconstructor), nt(x.nt) {
    // [fetalReconstruction] Clear addon
    addon.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    addon = 0;

    // [fetalReconstruction] Clear confidence map
    confidence_map.Initialize(
        reconstructor->_reconstructed.GetImageAttributes());
    confidence_map = 0;
  }

  void join(const ParallelSuperresolution &y) {
    addon += y.addon;
    confidence_map += y.confidence_map;
  }

    ParallelSuperresolution(irtkReconstructionEbb *reconstructor, int _nt, int _start, int _end)
	: reconstructor(reconstructor), nt(_nt), start(_start), end(_end) {
    // [fetalReconstruction] Clear addon
    addon.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    addon = 0;

    // [fetalReconstruction] Clear confidence map
    confidence_map.Initialize(
        reconstructor->_reconstructed.GetImageAttributes());
    confidence_map = 0;
  }

  void operator()() {
    task_scheduler_init init(nt);
    parallel_reduce(blocked_range<int>(0, (end-start)),
                    *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::Superresolution(int iter) {

  int i, j, k;
  irtkRealImage original;

  // [fetalReconstruction] Remember current reconstruction for edge-preserving smoothing
  original = _reconstructed;

#ifndef __EBBRT_BM__
  for (int i = 0; i < (int)nids.size(); i++) 
  {
      auto buf = MakeUniqueIOBuf(2 * sizeof(int));
      auto dp = buf->GetMutDataPointer();
      dp.Get<int>() = 8;
      dp.Get<int>() = iter;
      //FORPRINTF("Superresolution : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
      bytesTotal += buf->ComputeChainDataLength();
      SendMessage(nids[i], std::move(buf));
  }
  
  ParallelSuperresolution parallelSuperresolution(this, _numThreads, _start, _end);
  parallelSuperresolution();

  _addon = parallelSuperresolution.addon;
  _confidence_map = parallelSuperresolution.confidence_map;
  
  if (_debug) 
    FORPRINTF("[H] ParallelSuperResolution: Bloacking \n");
  testFuture = ebbrt::Promise<int>();
  auto tf = testFuture.GetFuture();
  tf.Block();
  if (_debug) 
    FORPRINTF("[H] ParallelSuperRedolution: Returning from future \n");
#else

  if(iter == 1)
  {
      _addon.Initialize(_reconstructed.GetImageAttributes());
      _confidence_map.Initialize(_reconstructed.GetImageAttributes());
  }

  // Clear addon
  _addon = 0;

  // Clear confidence map
  _confidence_map = 0;

  size_t ncpus = ebbrt::Cpu::Count();
  static ebbrt::SpinBarrier bar(ncpus);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t theCpu = ebbrt::Cpu::GetMine();
  int diff = (_end-_start);;
  for (size_t i = 0; i < ncpus; i++) {
      // spawn jobs on each core using SpawnRemote
      ebbrt::event_manager->SpawnRemote(
          [this, theCpu, ncpus, &count, &context, i, diff]() {
	      // get my cpu id
	      size_t mycpu = ebbrt::Cpu::GetMine();
	      int starte, ende, factor;
	      factor = (int)ceil(diff / (float)ncpus);
	      starte = (i * factor) + _start;
	      ende = (i * factor + factor) + _start;
	      ende = (ende > _end) ? _end : ende;

	      irtkRealImage maddon, m_confidence_map;
	      maddon.Initialize(_reconstructed.GetImageAttributes());
	      maddon = 0;
	      m_confidence_map.Initialize(_reconstructed.GetImageAttributes());
	      m_confidence_map = 0;
	      runSuperresolution(this, starte, ende, maddon, m_confidence_map);

	      // braces for scoping
	      {
		  // lock_guard auto unlocks end of scope, mutex doesn't work yet
		  std::lock_guard<ebbrt::SpinLock> l(spinlock);
		  this->_addon += maddon;
		  this->_confidence_map += m_confidence_map;
	      }

	      count++;
          if (_debug) 
	        FORPRINTF("[BM] Superresolution CPU %d barrier wait\n", mycpu);
	      bar.Wait();
          if (_debug) { 
	        FORPRINTF("[BM] Superresolution CPU %d barrier done\n", mycpu);
	        FORPRINTF("[BM] Superresolution CPU %d while wait\n", mycpu);
          }
	      while (count < (size_t)ncpus)
		  ;
          if (_debug) 
	        FORPRINTF("[BM] Superresolution CPU %d while done\n", mycpu);
	      if (mycpu == theCpu) {
		  ebbrt::event_manager->ActivateContext(std::move(context));
	      }
          },
          indexToCPU(
              i)); // if i don't add indexToCPU, one of the cores never run ? ?
  }
  ebbrt::event_manager->SaveContext(context);
  
  //FORPRINTF("runSuperresolution\n");
  //runSuperresolution(this, _start, _end, _addon, _confidence_map);
  return;
#endif
  
  //FORPRINTF("_addon = %lf, __confidence_map = %lf\n", _addon.Sum(), _confidence_map.Sum());
  
  if (!_adaptive)
    for (i = 0; i < _addon.GetX(); i++)
      for (j = 0; j < _addon.GetY(); j++)
        for (k = 0; k < _addon.GetZ(); k++)
          if (_confidence_map(i, j, k) > 0) {
            // [fetalReconstruction] ISSUES if _confidence_map(i, j, k) is too 
            // [fetalReconstruction] small leading to bright pixels
            _addon(i, j, k) /= _confidence_map(i, j, k);
            // [fetalReconstruction] this is to revert to normal (non-adaptive) 
            // [fetalReconstruction] regularisation
            _confidence_map(i, j, k) = 1;
          }

  _reconstructed += _addon * _alpha; //_average_volume_weight;

  // [fetalReconstruction] bound the intensities
  for (i = 0; i < (int)_reconstructed.GetX(); i++)
    for (j = 0; j < (int)_reconstructed.GetY(); j++)
      for (k = 0; k < (int)_reconstructed.GetZ(); k++) {
        if (_reconstructed(i, j, k) < _min_intensity * 0.9)
          _reconstructed(i, j, k) = _min_intensity * 0.9;
        if (_reconstructed(i, j, k) > _max_intensity * 1.1)
          _reconstructed(i, j, k) = _max_intensity * 1.1;
      }

  // [fetalReconstruction] Smooth the reconstructed image
  AdaptiveRegularization(iter, original);

  // [fetalReconstruction] Remove the bias in the reconstructed volume 
  // [fetalReconstruction] compared to previous iteration
  if (_global_bias_correction) {
    BiasCorrectVolume(original);
  }
  
  //FORPRINTF("%lf\n", SumRecon());
}

// [EbbRT] corresponds to the ParallelMStep in the original code
void runMStep(irtkReconstructionEbb *reconstructor, int start, int end,
              double &sigma, double &mix, double &num, double &min,
              double &max) {
  for (int inputIndex = start; inputIndex < end; ++inputIndex) {
    // [fetalReconstruction] read the current slice
    irtkRealImage slice = reconstructor->_slices[inputIndex];

    // [fetalReconstruction] alias the current weight image
    irtkRealImage &w = reconstructor->_weights[inputIndex];

    // [fetalReconstruction] alias the current bias image
    irtkRealImage &b = reconstructor->_bias[inputIndex];

    // [fetalReconstruction] identify scale factor
    double scale = reconstructor->_scale_cpu[inputIndex];

    // [fetalReconstruction] calculate error
    for (int i = 0; i < slice.GetX(); i++) {
      for (int j = 0; j < slice.GetY(); j++) {
        if (slice(i, j, 0) != -1) {
          // [fetalReconstruction] bias correct and scale the slice
          slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

          // [fetalReconstruction] otherwise the error has no meaning - 
          // [fetalReconstruction] it is equal to slice intensity
          if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {

            slice(i, j, 0) -=
                reconstructor->_simulated_slices[inputIndex](i, j, 0);

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
}

class ParallelMStep {
    irtkReconstructionEbb *reconstructor;
    int nt;
    int start;
    int end;
    
public:
  double sigma;
  double mix;
  double num;
  double min;
  double max;

  void operator()(const blocked_range<int> &r) {
    runMStep(reconstructor, r.begin(), r.end(), sigma, mix, num, min, max);
  }

  ParallelMStep(ParallelMStep &x, split)
      : reconstructor(x.reconstructor), nt(x.nt), start(x.start), end(x.end) {
    sigma = 0;
    mix = 0;
    num = 0;
    min = 0;
    max = 0;
  }

  void join(const ParallelMStep &y) {
    if (y.min < min)
      min = y.min;
    if (y.max > max)
      max = y.max;

    sigma += y.sigma;
    mix += y.mix;
    num += y.num;
  }

    ParallelMStep(irtkReconstructionEbb *reconstructor, int _nt, int _start, int _end)
      : reconstructor(reconstructor), nt(_nt), start(_start), end(_end) {
    sigma = 0;
    mix = 0;
    num = 0;
    min = voxel_limits<irtkRealPixel>::max();
    max = voxel_limits<irtkRealPixel>::min();
  }

  void operator()() {
    task_scheduler_init init(nt);
    parallel_reduce(blocked_range<int>(0, end-start),
                    *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::MStep(int iter) {
#ifndef __EBBRT_BM__
    for (int i = 0; i < (int)nids.size(); i++) 
    {
	auto buf = MakeUniqueIOBuf((2 * sizeof(int)));
	auto dp = buf->GetMutDataPointer();
	dp.Get<int>() = 10;
	dp.Get<int>() = iter;
	
	//FORPRINTF("MStep : Sending %d bytes\n", (int)buf->ComputeChainDataLength());
	bytesTotal += buf->ComputeChainDataLength();
	SendMessage(nids[i], std::move(buf));
    }
    
    ParallelMStep parallelMStep(this, _numThreads, _start, _end);
    parallelMStep();
  
    _msigma = parallelMStep.sigma;
    _mmix = parallelMStep.mix;
    _mnum = parallelMStep.num;
    _mmin = parallelMStep.min;
    _mmax = parallelMStep.max;

    if (_debug) 
      FORPRINTF("[H] MStep: Bloacking \n");
    testFuture = ebbrt::Promise<int>();
    auto tf = testFuture.GetFuture();
    tf.Block();
    if (_debug) 
      FORPRINTF("[H] MStep: Returning from future \n");
#else
  _msigma = 0;
  _mmix = 0;
  _mnum = 0;
  _mmin = voxel_limits<irtkRealPixel>::max();
  _mmax = voxel_limits<irtkRealPixel>::min();

  size_t ncpus = ebbrt::Cpu::Count();

  static ebbrt::SpinBarrier bar(ncpus);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t theCpu = ebbrt::Cpu::GetMine();
  int diff = (_end-_start);
  for (size_t i = 0; i < ncpus; i++) {
      // spawn jobs on each core using SpawnRemote
      ebbrt::event_manager->SpawnRemote(
          [this, theCpu, ncpus, &count, &context, i, diff]() {
	      // get my cpu id
	      size_t mycpu = ebbrt::Cpu::GetMine();
	      int starte, ende, factor;
	      factor = (int)ceil(diff / (float)ncpus);
	      starte = (i * factor) + _start;
	      ende = (i * factor + factor) + _start;
	      ende = (ende > _end) ? _end : ende;

	      double msigma = 0;
	      double mmix = 0;
	      double mnum = 0;
	      double mmin = voxel_limits<irtkRealPixel>::max();
	      double mmax = voxel_limits<irtkRealPixel>::min();

	      runMStep(this, starte, ende, msigma, mmix, mnum, mmin, mmax);

	      // braces for scoping
	      {
		  // lock_guard auto unlocks end of scope, mutex doesn't work yet
		  std::lock_guard<ebbrt::SpinLock> l(spinlock);
		  if (mmin < _mmin)
		      _mmin = mmin;
		  if (mmax > _mmax)
		      _mmax = mmax;
		  _msigma += msigma;
		  _mmix += mmix;
		  _mnum += mnum;
	      }

	      count++;
          if (_debug) 
	        FORPRINTF("[BM] MStep CPU %d barrier wait\n", mycpu);
	      bar.Wait();
          if (_debug) {
	        FORPRINTF("[BM] MStep CPU %d barrier done\n", mycpu);
	        FORPRINTF("[BM] MStep CPU %d while wait\n", mycpu);
          }
	      while (count < (size_t)ncpus)
		  ;
          if (_debug) 
            FORPRINTF("[BM] MStep CPU %d while done\n", mycpu);
	      if (mycpu == theCpu) {
		  ebbrt::event_manager->ActivateContext(std::move(context));
	      }
          },
          indexToCPU(
              i)); // if i don't add indexToCPU, one of the cores never run ? ?
  }
  ebbrt::event_manager->SaveContext(context);
//FORPRINTF("runMStep\n");
//runMStep(this, _start, _end, _msigma, _mmix, _mnum, _mmin, _mmax);
  
  return;
#endif

  if (_mmix > 0) {
    _sigma_cpu = _msigma / _mmix;
  } else {
    //FORPRINTF("Something went wrong: sigma= %fmix=%f\n", _msigma, _mmix);
    exit(1);
  }
  if (_sigma_cpu < _step * _step / 6.28)
    _sigma_cpu = _step * _step / 6.28;

  if (iter > 1)
    _mix_cpu = _mmix / _mnum;

  // Calculate m
  _m_cpu = 1 / (_mmax - _mmin);

  //FORPRINTF("%lf %lf %lf %lf %lf %lf %lf %lf\n", _msigma, _mmix, _mnum, _mmin, _mmax, _sigma_cpu, _mix_cpu, _m_cpu);
}

void irtkReconstructionEbb::BiasCorrectVolume(irtkRealImage &original) {
    //FORPRINTF("In BiasCorrectVolume\n");
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
    if ((*pw == 1) && (*po > _low_intensity_cutoff * _max_intensity) &&
        (*pr > _low_intensity_cutoff * _max_intensity)) {
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
  irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
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
      if (*pi < _min_intensity * 0.9)
        *pi = _min_intensity * 0.9;
      if (*pi > _max_intensity * 1.1)
        *pi = _max_intensity * 1.1;
    } else {
      *pr = 0;
    }
    pr++;
    pw++;
    pm++;
    pi++;
  }
}

void irtkReconstructionEbb::Evaluate(int iter) {
  int sum = 0;
  unsigned int i;
  for (i = 0; i < _slices.size(); i++) {
    if ((_slice_weight_cpu[i] >= 0.5) && (_slice_inside_cpu[i])) {
      sum++;
    }
  }

  sum = 0;
  for (i = 0; i < _slices.size(); i++) {
    if ((_slice_weight_cpu[i] < 0.5) && (_slice_inside_cpu[i])) {
      sum++;
    }
  }

  sum = 0;
  for (i = 0; i < _slices.size(); i++) {
    if (!(_slice_inside_cpu[i])) {
      sum++;
    }
  }
}

void irtkReconstructionEbb::NormaliseBias(int iter) {
  irtkRealImage bias;
  bias.Initialize(_reconstructed.GetImageAttributes());
  bias = 0;

  for (size_t inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
    // alias the current slice
    irtkRealImage &slice = _slices[inputIndex];

    // read the current bias image
    irtkRealImage b = _bias[inputIndex];

    // read current scale factor
    double scale = _scale_cpu[inputIndex];

    irtkRealPixel *pi = slice.GetPointerToVoxels();
    irtkRealPixel *pb = b.GetPointerToVoxels();
    for (int i = 0; i < slice.GetNumberOfVoxels(); i++) {
      if ((*pi > -1) && (scale > 0))
        *pb -= log(scale);
      pb++;
      pi++;
    }

    // Distribute slice intensities to the volume
    POINT3D p;
    for (int i = 0; i < slice.GetX(); i++)
      for (int j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // number of volume voxels with non-zero coefficients for current
          // slice voxel
          int n = _volcoeffs[inputIndex][i][j].size();
          // add contribution of current slice voxel to all voxel volumes
          // to which it contributes
          for (int k = 0; k < n; k++) {
            p = _volcoeffs[inputIndex][i][j][k];
            bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
          }
        }
  } // end of loop for a slice inputIndex

  // [fetalReconstruction] normalize the volume by proportion of contributing 
  // [fetalReconstruction] slice voxels for each volume voxel
  bias /= _volume_weights;

  MaskImage(bias, 0);
  irtkRealImage m = _mask;
  irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
  gb.SetInput(&bias);
  gb.SetOutput(&bias);
  gb.Run();
  gb.SetInput(&m);
  gb.SetOutput(&m);
  gb.Run();
  bias /= m;

  irtkRealPixel *pi, *pb;
  pi = _reconstructed.GetPointerToVoxels();
  pb = bias.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*pi != -1)
      *pi /= exp(-(*pb));
    pi++;
    pb++;
  }
}

void irtkReconstructionEbb::ReadTransformation(char *folder) {
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
  _transformations_gpu.clear();
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
    _transformations_gpu.push_back(*rigidTransf);
    delete transformation;
    cout << path << endl;
  }
}

void irtkReconstructionEbb::SetReconstructed(irtkRealImage &reconstructed) {
  _reconstructed = reconstructed;
  _template_created = true;
}

void irtkReconstructionEbb::SetTransformations(
    vector<irtkRigidTransformation> &transformations) {
  _transformations.clear();
  _transformations_gpu.clear();
  for (int i = 0; i < transformations.size(); i++) {
    _transformations.push_back(transformations[i]);
    _transformations_gpu.push_back(transformations[i]);
  }
}

void irtkReconstructionEbb::SaveBiasFields() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "bias%i.nii.gz", inputIndex);
    _bias[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveConfidenceMap() {
  _confidence_map.Write("confidence-map.nii.gz");
}

void irtkReconstructionEbb::SaveSlices() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "slice%i.nii.gz", inputIndex);
    _slices[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveWeights() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "weights%i.nii.gz", inputIndex);
    _weights[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveTransformations() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "transformation%i.dof", inputIndex);
    _transformations[inputIndex].irtkTransformation::Write(buffer);
    sprintf(buffer, "transformation_gpu%i.dof", inputIndex);
    _transformations_gpu[inputIndex].irtkTransformation::Write(buffer);
  }
}

void irtkReconstructionEbb::GetTransformations(
    vector<irtkRigidTransformation> &transformations) {
  transformations.clear();
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    transformations.push_back(_transformations[inputIndex]);
  }
}

void irtkReconstructionEbb::GetSlices(vector<irtkRealImage> &slices) {
  slices.clear();
  for (unsigned int i = 0; i < _slices.size(); i++)
    slices.push_back(_slices[i]);
}

void irtkReconstructionEbb::SlicesInfo(const char *filename) {
  ofstream info;
  info.open(filename);

  // header
  info << "stack_index"
       << "\t"
       << "included"
       << "\t" // Included slices
       << "excluded"
       << "\t" // Excluded slices
       << "outside"
       << "\t" // Outside slices
       << "weight"
       << "\t"
       << "scale"
       << "\t"
       << "TranslationX"
       << "\t"
       << "TranslationY"
       << "\t"
       << "TranslationZ"
       << "\t"
       << "RotationX"
       << "\t"
       << "RotationY"
       << "\t"
       << "RotationZ" << endl;

  for (int i = 0; i < _slices.size(); i++) {
    irtkRigidTransformation &t = _transformations[i];
    info << _stack_index[i] << "\t"
         << (((_slice_weight_cpu[i] >= 0.5) && (_slice_inside_cpu[i])) ? 1 : 0)
         << "\t" // Included slices
         << (((_slice_weight_cpu[i] < 0.5) && (_slice_inside_cpu[i])) ? 1 : 0)
         << "\t"                                        // Excluded slices
         << ((!(_slice_inside_cpu[i])) ? 1 : 0) << "\t" // Outside slices
         << _slice_weight_cpu[i] << "\t" << _scale_cpu[i] << "\t"
         << t.GetTranslationX() << "\t" << t.GetTranslationY() << "\t"
         << t.GetTranslationZ() << "\t" << t.GetRotationX() << "\t"
         << t.GetRotationY() << "\t" << t.GetRotationZ() << endl;
  }

  info.close();
}

void irtkReconstructionEbb::SplitImage(irtkRealImage image, int packages,
                                       vector<irtkRealImage> &stacks) {
  irtkImageAttributes attr = image.GetImageAttributes();

  // [fetalReconstruction] slices in package
  int pkg_z = attr._z / packages;
  double pkg_dz = attr._dz * packages;
  cout << "packages: " << packages << "; slices: " << attr._z
       << "; slices in package: " << pkg_z << endl;
  cout << "slice thickness " << attr._dz
       << "; slickess thickness in package: " << pkg_dz << endl;

  char buffer[256];
  int i, j, k, l;
  double x, y, z, sx, sy, sz, ox, oy, oz;
  for (l = 0; l < packages; l++) {
    attr = image.GetImageAttributes();
    if ((pkg_z * packages + l) < attr._z)
      attr._z = pkg_z + 1;
    else
      attr._z = pkg_z;
    attr._dz = pkg_dz;

    cout << "split image " << l << " has " << attr._z << " slices." << endl;

    // [fetalReconstruction] fill values in each stack
    irtkRealImage stack(attr);
    stack.GetOrigin(ox, oy, oz);

    cout << "Stack " << l << ":" << endl;
    for (k = 0; k < stack.GetZ(); k++)
      for (j = 0; j < stack.GetY(); j++)
        for (i = 0; i < stack.GetX(); i++)
          stack.Put(i, j, k, image(i, j, k * packages + l));

    // [fetalReconstruction] adjust origin

    // [fetalReconstruction] original image coordinates
    x = 0;
    y = 0;
    z = l;
    image.ImageToWorld(x, y, z);
    cout << "image: " << x << " " << y << " " << z << endl;
    // [fetalReconstruction] stack coordinates
    sx = 0;
    sy = 0;
    sz = 0;
    stack.PutOrigin(ox, oy, oz); // [fetalReconstruction] adjust to original value
    stack.ImageToWorld(sx, sy, sz);
    cout << "stack: " << sx << " " << sy << " " << sz << endl;
    // [fetalReconstruction] adjust origin
    cout << "adjustment needed: " << x - sx << " " << y - sy << " " << z - sz
         << endl;
    stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
    sx = 0;
    sy = 0;
    sz = 0;
    stack.ImageToWorld(sx, sy, sz);
    cout << "adjusted: " << sx << " " << sy << " " << sz << endl;

    stacks.push_back(stack);
  }
}

void irtkReconstructionEbb::SplitImageEvenOdd(irtkRealImage image, int packages,
                                              vector<irtkRealImage> &stacks) {
  vector<irtkRealImage> packs;
  vector<irtkRealImage> packs2;
  cout << "Split Image Even Odd: " << packages << " packages." << endl;

  stacks.clear();
  SplitImage(image, packages, packs);
  for (int i = 0; i < packs.size(); i++) {
    cout << "Package " << i << ": " << endl;
    packs2.clear();
    SplitImage(packs[i], 2, packs2);
    stacks.push_back(packs2[0]);
    stacks.push_back(packs2[1]);
  }
}

void irtkReconstructionEbb::SplitImageEvenOddHalf(irtkRealImage image,
                                                  int packages,
                                                  vector<irtkRealImage> &stacks,
                                                  int iter) {
  vector<irtkRealImage> packs;
  vector<irtkRealImage> packs2;

  cout << "Split Image Even Odd Half " << iter << endl;
  stacks.clear();
  if (iter > 1)
    SplitImageEvenOddHalf(image, packages, packs, iter - 1);
  else
    SplitImageEvenOdd(image, packages, packs);
  for (int i = 0; i < packs.size(); i++) {
    packs2.clear();
    HalfImage(packs[i], packs2);
    for (int j = 0; j < packs2.size(); j++)
      stacks.push_back(packs2[j]);
  }
}

void irtkReconstructionEbb::HalfImage(irtkRealImage image,
                                      vector<irtkRealImage> &stacks) {
  irtkRealImage tmp;
  irtkImageAttributes attr = image.GetImageAttributes();
  stacks.clear();

  // [fetalReconstruction] We would not like single slices - that is reserved 
  // [fetalReconstruction] for slice-to-volume
  if (attr._z >= 4) {
    tmp = image.GetRegion(0, 0, 0, attr._x, attr._y, attr._z / 2);
    stacks.push_back(tmp);
    tmp = image.GetRegion(0, 0, attr._z / 2, attr._x, attr._y, attr._z);
    stacks.push_back(tmp);
  } else
    stacks.push_back(image);
}

void irtkReconstructionEbb::PackageToVolume(vector<irtkRealImage> &stacks,
                                            vector<int> &pack_num, bool evenodd,
                                            bool half, int half_iter) {
  irtkImageRigidRegistrationWithPadding rigidregistration;
  irtkGreyImage t, s;
  vector<irtkRealImage> packages;
  char buffer[256];

  int firstSlice = 0;
  cout << "Package to volume: " << endl;
  for (unsigned int i = 0; i < stacks.size(); i++) {
    cout << "Stack " << i << ": First slice index is " << firstSlice << endl;

    packages.clear();
    if (evenodd) {
      if (half)
        SplitImageEvenOddHalf(stacks[i], pack_num[i], packages, half_iter);
      else
        SplitImageEvenOdd(stacks[i], pack_num[i], packages);
    } else
      SplitImage(stacks[i], pack_num[i], packages);

    for (unsigned int j = 0; j < packages.size(); j++) {
      cout << "Package " << j << " of stack " << i << endl;
      if (_debug) {
        sprintf(buffer, "package%i-%i.nii.gz", i, j);
        packages[j].Write(buffer);
      }

      t = packages[j];
      s = _reconstructed;

      // [fetalReconstruction] find existing transformation
      double x, y, z;
      x = 0;
      y = 0;
      z = 0;
      packages[j].ImageToWorld(x, y, z);
      stacks[i].WorldToImage(x, y, z);

      int firstSliceIndex = round(z) + firstSlice;
      cout << "First slice index for package " << j << " of stack " << i
           << " is " << firstSliceIndex << endl;

      // [fetalReconstruction] put origin in target to zero
      irtkRigidTransformation offset;
      ResetOrigin(t, offset);
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = _transformations[firstSliceIndex].GetMatrix();
      m = m * mo;
      _transformations[firstSliceIndex].PutMatrix(m);

      rigidregistration.SetInput(&t, &s);
      rigidregistration.SetOutput(&_transformations[firstSliceIndex]);
      rigidregistration.GuessParameterSliceToVolume();
      if (_debug) {
        sprintf(buffer, "par-packages.rreg");
        rigidregistration.Write(buffer);
      }
      rigidregistration.Run();

      // [fetalReconstruction] undo the offset
      mo.Invert();
      m = _transformations[firstSliceIndex].GetMatrix();
      m = m * mo;
      _transformations[firstSliceIndex].PutMatrix(m);

      if (_debug) {
        sprintf(buffer, "transformation%i-%i.dof", i, j);
        _transformations[firstSliceIndex].irtkTransformation::Write(buffer);
      }

      // [fetalReconstruction] set the transformation to all slices of the package
      cout << "Slices of the package " << j << " of the stack " << i
           << " are: ";
      for (int k = 0; k < packages[j].GetZ(); k++) {
        x = 0;
        y = 0;
        z = k;
        packages[j].ImageToWorld(x, y, z);
        stacks[i].WorldToImage(x, y, z);
        int sliceIndex = round(z) + firstSlice;
        cout << sliceIndex << " " << endl;

        if (sliceIndex >= _transformations.size()) {
          cerr << "irtkRecnstruction::PackageToVolume: sliceIndex out of range."
               << endl;
          cerr << sliceIndex << " " << _transformations.size() << endl;
          exit(1);
        }

        if (sliceIndex != firstSliceIndex) {
          _transformations[sliceIndex].PutTranslationX(
              _transformations[firstSliceIndex].GetTranslationX());
          _transformations[sliceIndex].PutTranslationY(
              _transformations[firstSliceIndex].GetTranslationY());
          _transformations[sliceIndex].PutTranslationZ(
              _transformations[firstSliceIndex].GetTranslationZ());
          _transformations[sliceIndex].PutRotationX(
              _transformations[firstSliceIndex].GetRotationX());
          _transformations[sliceIndex].PutRotationY(
              _transformations[firstSliceIndex].GetRotationY());
          _transformations[sliceIndex].PutRotationZ(
              _transformations[firstSliceIndex].GetRotationZ());
          _transformations[sliceIndex].UpdateMatrix();
        }
      }
    }
    firstSlice += stacks[i].GetZ();
  }
}

void irtkReconstructionEbb::CropImage(irtkRealImage &image,
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

void irtkReconstructionEbb::InvertStackTransformations(
    vector<irtkRigidTransformation> &stack_transformations) {
  for (unsigned int i = 0; i < stack_transformations.size(); i++) {
    // [fetalReconstruction] invert transformation for the stacks
    stack_transformations[i].Invert();
    stack_transformations[i].UpdateParameter();
  }
}

void irtkReconstructionEbb::MaskVolume() {
  irtkRealPixel *pr = _reconstructed.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*pm == 0)
      *pr = -1;
    pm++;
    pr++;
  }
  //FORPRINTF("MaskVolume : %lf %lf\n", _reconstructed.Sum(), _mask.Sum());
}

void irtkReconstructionEbb::MaskImage(irtkRealImage &image, double padding) {
  if (image.GetNumberOfVoxels() != _mask.GetNumberOfVoxels()) {
    cerr << "Cannot mask the image - different dimensions" << endl;
    exit(1);
  }
  irtkRealPixel *pr = image.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
    if (*pm == 0)
      *pr = padding;
    pm++;
    pr++;
  }
}

/// [fetalReconstruction] Like PutMinMax but ignoring negative values (mask)
void irtkReconstructionEbb::Rescale(irtkRealImage &img, double max) {
  int i, n;
  irtkRealPixel *ptr, min_val, max_val;

  // [fetalReconstruction] Get lower and upper bound
  img.GetMinMax(&min_val, &max_val);

  n = img.GetNumberOfVoxels();
  ptr = img.GetPointerToVoxels();
  for (i = 0; i < n; i++)
    if (ptr[i] > 0)
      ptr[i] = double(ptr[i]) / double(max_val) * max;
}

void irtkReconstructionEbb::RunRecon(int iterations, double delta,
                                     double lastIterLambda,
                                     int rec_iterations_first,
                                     int rec_iterations_last,
                                     bool intensity_matching, double lambda,
                                     int levels) {
  int i;
  int rec_iterations;
  float timers[13] = {0};
  struct timeval tstart, tend;
  struct timeval lstart, lend;
  float sumCompute = 0.0;
  float tempTime = 0.0;
  float temp2 = 0.0;

  //FORPRINTF("irtkReconstructionEbb::RunRecon()\n");

  gettimeofday(&tstart, NULL);

  //FORPRINTF("restoresliceintensities = %lf %d\n", sumFVec(_stack_factor), sumInt(_stack_index));
  for (int iter = 0; iter < iterations; iter++) {
      FORPRINTF("iter = %d\n", iter);
    // perform slice-to-volume registrations - skip the first iteration
    if (iter > 0) {
      gettimeofday(&lstart, NULL);
      SliceToVolumeRegistration();
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SLICETOVOLUMEREGISTRATION] += tempTime;
      sumCompute += tempTime;
    }

    if (iter == (iterations - 1)) {
      SetSmoothingParameters(delta, lastIterLambda);
    } else {
      double l = lambda;
      for (i = 0; i < levels; i++) {
        if (iter == iterations * (levels - i - 1) / levels) {
          SetSmoothingParameters(delta, l);
        }
        l *= 2;
      }
    }

    // Use faster reconstruction during iterations and slower for final
    // reconstruction
    if (iter < (iterations - 1)) {
      SpeedupOn();
    } else {
      SpeedupOff();
    }

    // Initialise values of weights, scales and bias fields
    gettimeofday(&lstart, NULL);
    InitializeEMValues();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[INITIALIZEEMVALUES] += tempTime;
    sumCompute += tempTime;

    // Calculate matrix of transformation between voxels of slices and volume
    gettimeofday(&lstart, NULL);
    CoeffInit(iter);
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[COEFFINIT] += tempTime;
    sumCompute += tempTime;
    //goto done;
    
    // Initialize reconstructed image with Gaussian weighted reconstruction
    gettimeofday(&lstart, NULL);
    GaussianReconstruction();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[GAUSSIANRECONSTRUCTION] += tempTime;
    sumCompute += tempTime;
    //if (iter > 0) goto done;
    
    // Simulate slices (needs to be done after Gaussian reconstruction)
    gettimeofday(&lstart, NULL);
    if (iter == 0) SimulateSlices(false);
    else SimulateSlices(true);
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[SIMULATESLICES] += tempTime;
    sumCompute += tempTime;

    gettimeofday(&lstart, NULL);
    InitializeRobustStatistics();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[INITIALIZEROBUSTSTATISTICS] += tempTime;
    sumCompute += tempTime;
    
    gettimeofday(&lstart, NULL);
    EStep();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[ESTEP] += tempTime;
    sumCompute += tempTime;
    
    // number of reconstruction iterations
    if (iter == (iterations - 1)) {
      rec_iterations = rec_iterations_last;
    } else
      rec_iterations = rec_iterations_first;

    // reconstruction iterations
    i = 0;
    for (i = 0; i < rec_iterations; i++) {
	//FORPRINTF("rec_iterations %d\n", i);

      if (intensity_matching) {
        // calculate bias fields
        gettimeofday(&lstart, NULL);
        // calculate scales
        Scale();
        gettimeofday(&lend, NULL);
        tempTime = (lend.tv_sec - lstart.tv_sec) +
                   ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
        timers[SCALE] += tempTime;
        sumCompute += tempTime;
	
	//if(i == 1) goto done;
      }

      // MStep and update reconstructed volume
      gettimeofday(&lstart, NULL);
      Superresolution(i + 1);
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SUPERRESOLUTION] += tempTime;
      sumCompute += tempTime;
      
      // Simulate slices (needs to be done
      // after the update of the reconstructed volume)
      gettimeofday(&lstart, NULL);
      SimulateSlices(true);
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SIMULATESLICES] += tempTime;
      sumCompute += tempTime;
	    
      gettimeofday(&lstart, NULL);
      MStep(i + 1);
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[MSTEP] += tempTime;
      sumCompute += tempTime;
      
      gettimeofday(&lstart, NULL);
      EStep();
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[ESTEP] += tempTime;
      sumCompute += tempTime;
    } // end of reconstruction iterations
    
    // Mask reconstructed image to ROI given by the mask
    gettimeofday(&lstart, NULL);
    MaskVolume();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[MASKVOLUME] += tempTime;
    sumCompute += tempTime;

  } // end of interleaved registration-reconstruction iterations

  gettimeofday(&lstart, NULL);
  RestoreSliceIntensities();
  ScaleVolume();
  gettimeofday(&lend, NULL);
  tempTime = (lend.tv_sec - lstart.tv_sec) +
             ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
  timers[RESTORESLICE] += tempTime;
  sumCompute += tempTime;

  gettimeofday(&tend, NULL);

  if (_debug) { 
  FORPRINTF("SliceToVolumeRegistration: %lf seconds\n",
            timers[SLICETOVOLUMEREGISTRATION]);
  FORPRINTF("InitializeEMValues: %lf seconds\n", timers[INITIALIZEEMVALUES]);
  FORPRINTF("CoeffInit: %lf seconds\n", timers[COEFFINIT]);
  FORPRINTF("GaussianReconstruction: %lf seconds\n",
            timers[GAUSSIANRECONSTRUCTION]);
  FORPRINTF("SimulateSlices: %lf seconds\n", timers[SIMULATESLICES]);
  FORPRINTF("InitializeRobustStatistics: %lf seconds\n",
            timers[INITIALIZEROBUSTSTATISTICS]);
  FORPRINTF("EStep: %lf seconds\n", timers[ESTEP]);
  FORPRINTF("Scale: %lf seconds\n", timers[SCALE]);
  FORPRINTF("Superresolution: %lf seconds\n", timers[SUPERRESOLUTION]);
  FORPRINTF("MStep: %lf seconds\n", timers[MSTEP]);
  FORPRINTF("MaskVolume: %lf seconds\n", timers[MASKVOLUME]);
  //FORPRINTF("Evaluate: %lf seconds\n", timers[EVALUATE]);
  FORPRINTF("RestoreSliceIntensities and ScaleVolume: %lf seconds\n",
            timers[RESTORESLICE]);

  FORPRINTF("compute time: %lf seconds\n",
            (tend.tv_sec - tstart.tv_sec) +
                ((tend.tv_usec - tstart.tv_usec) / 1000000.0));
  FORPRINTF("sumCompute: %lf seconds\n", sumCompute);


  for (i = 0; i < 13; i++) {
    temp2 += timers[i];
  }
  FORPRINTF("sumTimers: %lf seconds\n", temp2);

  FORPRINTF("checksum _reconstructed = %lf\n", SumRecon());

  FORPRINTF("\nTotal Bytes Sent: %ld\n", bytesTotal);
  }
done:
  mypromise.SetValue();
  // save final result
  //_reconstructed.Write("3TStackReconstruction.nii");
}
// [SMAFJAS] THIS IS CALL IMPLICITY EVERY TIME AN IO EVENT OCCUR (EBBRT)
// [SMAFJAS] REQUIRE WHEN INHERIT MESSAGABLE
void irtkReconstructionEbb::ReceiveMessage(Messenger::NetworkId nid,
                                           std::unique_ptr<IOBuf> &&buffer) {
  auto dp = buffer->GetDataPointer();
  auto ret = dp.Get<int>();// [SMAFJAS] THIS IS THE FUNCTION 
#ifdef __EBBRT_BM__
  if (_debug) 
    FORPRINTF("[BM] Received %d bytes, %d\n", (int)buffer->ComputeChainDataLength(), ret);
#else
  if (_debug) 
    FORPRINTF("[H] Received %d bytes, %d\n", (int)buffer->ComputeChainDataLength(), ret);
#endif
  bytesTotal += buffer->ComputeChainDataLength();
  
// backend
#ifdef __EBBRT_BM__
  if (ret == 0) // [SMAFJAS] COEFF INIT FUNCTION
  {
      nids.clear();
      nids.push_back(nid);
      
      _start = dp.Get<int>();
      _end = dp.Get<int>();

      _delta = dp.Get<double>();
      _lambda = dp.Get<double>();
      _alpha = dp.Get<double>();
      _quality_factor = dp.Get<double>();
      
      int sfs = dp.Get<int>();
      int sis = dp.Get<int>();
      
      auto nslices = dp.Get<int>();
      _slices.resize(nslices);
    
      //FORPRINTF("EBBRT BM received %d, nslices = %d\n", ret, nslices);

      for(int i = 0; i < nslices; i++)
      {
	  DeserializeSlice(dp, _slices[i]);
      }      
      DeserializeSlice(dp, _reconstructed);      
      DeserializeSlice(dp, _mask);
      auto nrigidtrans = dp.Get<int>();	
      _transformations.resize(nrigidtrans);
      
      for(int i = 0; i < nrigidtrans; i++)
      {
	  DeserializeTransformations(dp, _transformations[i]);
      }

      _stack_factor.resize(sfs);
      dp.Get(sfs*sizeof(float), (uint8_t*)_stack_factor.data());
      
      _stack_index.resize(sis);
      dp.Get(sis*sizeof(int), (uint8_t*)_stack_index.data());
      
      _global_bias_correction = false;
      _step = 0.0001;
      _sigma_bias = 12;
      _sigma_s_cpu = 0.025;
      _sigma_s2_cpu = 0.025;
      _mix_s_cpu = 0.9;
      _mix_cpu = 0.9;
      _low_intensity_cutoff = 0.01;
      _adaptive = false;
      // [SMAFJAS] CAN WE PUT THIS IN THE HEADER FILE?
      int directions[13][3] = {{1, 0, -1}, {0, 1, -1}, {1, 1, -1}, {1, -1,
								    -1},
			       {1, 0, 0},  {0, 1, 0},  {1, 1, 0},  {1, -1, 0},
			       {1, 0, 1},  {0, 1, 1},  {1, 1, 1},  {1, -1, 1},
			       {0, 0, 1}};
      for (int i = 0; i < 13; i++)
	  for (int j = 0; j < 3; j++)
	      _directions[i][j] = directions[i][j];

      InitializeEM();
      InitializeEMValues();
      CoeffInit(0);
      // [SMAFJAS] SEND DATA TO THE FRONTEND TO NOTIFY WORK IS DONE
      // sending
      auto retbuf = MakeUniqueIOBuf(1 * sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      SendMessage(nids[0], std::move(retbuf));
  } 
  else if(ret == 1)
  {
     GaussianReconstruction();
  }
  else if(ret == 2)
  {
      _simulated_slices.clear();
      _simulated_weights.clear();
      _simulated_inside.clear();
      
      int i;
      for(i=0;i<_slices.size();i++)
      {
	  _simulated_slices.push_back(_slices[i]);
	  _simulated_weights.push_back(_slices[i]);
	  _simulated_inside.push_back(_slices[i]);
      }

      int reconSize = dp.Get<int>();
      dp.Get(reconSize*sizeof(double), (uint8_t*)_reconstructed.GetMat());
      
      SimulateSlices(false);
	    
      // sending
      auto retbuf = MakeUniqueIOBuf(1 * sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 3)
  {
      InitializeRobustStatistics();
  }
  else if(ret == 4)
  {
      _sigma_cpu = dp.Get<double>();
      _sigma_s_cpu = dp.Get<double>();
      _mix_cpu = dp.Get<double>();
      _mix_s_cpu = dp.Get<double>();
      _m_cpu = dp.Get<double>();
      
      int tmp = dp.Get<int>();
      _small_slices.resize(tmp);
      dp.Get(tmp*sizeof(int), (uint8_t*)_small_slices.data());
      
      EStep();
  }
  else if (ret == 5)
  {
      _mean_s_cpu = dp.Get<double>();
      _mean_s2_cpu = dp.Get<double>();
      
      _ttsum = 0;
      _ttden = 0;
      _ttsum2 = 0;
      _ttden2 = 0;

      for (int inputIndex = _start; inputIndex < _end; inputIndex++)
      {
	  if (slice_potential[inputIndex] >= 0) 
	  {
	      _ttsum += (slice_potential[inputIndex] - _mean_s_cpu) *
		  (slice_potential[inputIndex] - _mean_s_cpu) *
		  _slice_weight_cpu[inputIndex];
	  
	      _ttden += _slice_weight_cpu[inputIndex];
	  
	      _ttsum2 += (slice_potential[inputIndex] - _mean_s2_cpu) *
		  (slice_potential[inputIndex] - _mean_s2_cpu) *
		  (1 - _slice_weight_cpu[inputIndex]);
	  
	      _ttden2 += (1 - _slice_weight_cpu[inputIndex]);
	  }
      }

      // sending
      auto retbuf = MakeUniqueIOBuf((1*sizeof(int)) + (4 * sizeof(double)));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = 5;
      retdp.Get<double>() = _ttsum;
      retdp.Get<double>() = _ttden;
      retdp.Get<double>() = _ttsum2;
      retdp.Get<double>() = _ttden2;
      SendMessage(nids[0], std::move(retbuf));

  }
  else if(ret == 6)
  {
      _sigma_s_cpu = dp.Get<double>();
      _sigma_s2_cpu = dp.Get<double>();
      
      _ttsum = 0;
      _ttnum = 0;
      double gs1, gs2;
      
      for (int inputIndex = _start; inputIndex < _end; inputIndex++) {
	  // [fetalReconstruction] Slice does not have any voxels in volumetric ROI
	  if (slice_potential[inputIndex] == -1) {
	      _slice_weight_cpu[inputIndex] = 0;
	      continue;
	  }

	  // [fetalReconstruction] All slices are outliers or the means are not valid
	  if ((_ttden <= 0) || (_mean_s2_cpu <= _mean_s_cpu)) {
	      _slice_weight_cpu[inputIndex] = 1;
	      continue;
	  }

	  // [fetalReconstruction] likelihood for inliers
	  if (slice_potential[inputIndex] < _mean_s2_cpu)
	      gs1 = G(slice_potential[inputIndex] - _mean_s_cpu, _sigma_s_cpu);
	  else
	      gs1 = 0;

	  // [fetalReconstruction] likelihood for outliers
	  if (slice_potential[inputIndex] > _mean_s_cpu)
	      gs2 = G(slice_potential[inputIndex] - _mean_s2_cpu, _sigma_s2_cpu);
	  else
	      gs2 = 0;

	  // [fetalReconstruction] calculate slice weight
	  double likelihood = gs1 * _mix_s_cpu + gs2 * (1 - _mix_s_cpu);
	  if (likelihood > 0)
	      _slice_weight_cpu[inputIndex] = gs1 * _mix_s_cpu / likelihood;
	  else {
	      if (slice_potential[inputIndex] <= _mean_s_cpu)
		  _slice_weight_cpu[inputIndex] = 1;
	      if (slice_potential[inputIndex] >= _mean_s2_cpu)
		  _slice_weight_cpu[inputIndex] = 0;
	      if ((slice_potential[inputIndex] < _mean_s2_cpu) &&
		  (slice_potential[inputIndex] > _mean_s_cpu)) // should not happen
		  _slice_weight_cpu[inputIndex] = 1;
	  }

	  if (slice_potential[inputIndex] >= 0) {
	      _ttsum += _slice_weight_cpu[inputIndex];
	      _ttnum ++;
	  }
      }
      
      auto retbuf = MakeUniqueIOBuf((2*sizeof(int)) + (1 * sizeof(double)));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = 6;
      retdp.Get<double>() = _ttsum;
      retdp.Get<int>() = _ttnum;
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 7)
  {
      Scale();
      
      auto retbuf = MakeUniqueIOBuf(1*sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 8)
  {
      int ite = dp.Get<int>();
      Superresolution(ite);

      auto retbuf = MakeUniqueIOBuf(1*sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = 8;
      retbuf->PrependChain(std::move(serializeSlices(_addon)));
      retbuf->PrependChain(std::move(serializeSlices(_confidence_map)));
      if (_debug) 
        FORPRINTF("[BM] ret 8 - Superresolution : Sending %d bytes\n", (int)retbuf->ComputeChainDataLength());
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 9)
  {
      int reconSize = dp.Get<int>();
      dp.Get(reconSize*sizeof(double), (uint8_t*)_reconstructed.GetMat());
      SimulateSlices(true);

      // sending
      auto retbuf = MakeUniqueIOBuf(1 * sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = 2;
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 10)
  {
      int ite = dp.Get<int>();
      MStep(ite);

      auto retbuf = MakeUniqueIOBuf((5 * sizeof(double)) + (1 * sizeof(int)));
      auto retdp = retbuf->GetMutDataPointer();
      
      retdp.Get<int>() = ret;
      retdp.Get<double>() = _msigma;
      retdp.Get<double>() = _mmix; 
      retdp.Get<double>() = _mnum; 
      retdp.Get<double>() = _mmin; 
      retdp.Get<double>() = _mmax;

      // FORPRINTF("MStep : Sending %d bytes\n", (int)retbuf->ComputeChainDataLength());
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 11)
  {
      RestoreSliceIntensities();
      auto retbuf = MakeUniqueIOBuf((1 * sizeof(int)));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 12)
  {
      ScaleVolume();
      
      auto retbuf = MakeUniqueIOBuf((1 * sizeof(int)) + (2 * sizeof(double)));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      retdp.Get<double>() = _sscalenum;
      retdp.Get<double>() = _sscaleden;
      
      SendMessage(nids[0], std::move(retbuf));
  }
  else if(ret == 13)
  {
      int reconSize = dp.Get<int>();
      dp.Get(reconSize*sizeof(double), (uint8_t*)_reconstructed.GetMat());
      
      SliceToVolumeRegistration();

      auto retbuf = MakeUniqueIOBuf((3 * sizeof(int)));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      retdp.Get<int>() = _start;
      retdp.Get<int>() = _end;
      
      for(int i = _start; i < _end; i++)
      {
	  retbuf->PrependChain(std::move(serializeRigidTrans(_transformations[i])));
      }
	  
      SendMessage(nids[0], std::move(retbuf));
  }
  else if (ret == 14)
  {
      _delta = dp.Get<double>();
      _lambda = dp.Get<double>();
      _alpha = dp.Get<double>();
      _quality_factor = dp.Get<double>();

      auto nrigidtrans = dp.Get<int>();
      // FORPRINTF("nrigidtrans = %d\n", nrigidtrans);
      
      for(int i = 0; i < nrigidtrans; i++)
      {
	  _transformations[i]._tx = dp.Get<double>();
	  _transformations[i]._ty = dp.Get<double>();
	  _transformations[i]._tz = dp.Get<double>();

	  _transformations[i]._rx = dp.Get<double>();
	  _transformations[i]._ry = dp.Get<double>();
	  _transformations[i]._rz = dp.Get<double>();

	  _transformations[i]._cosrx = dp.Get<double>();
	  _transformations[i]._cosry = dp.Get<double>();
	  _transformations[i]._cosrz = dp.Get<double>();

	  _transformations[i]._sinrx = dp.Get<double>();
	  _transformations[i]._sinry = dp.Get<double>();
	  _transformations[i]._sinrz = dp.Get<double>();
	
	  _transformations[i]._status[0] = (_Status) dp.Get<int>();
	  _transformations[i]._status[1] = (_Status) dp.Get<int>();
	  _transformations[i]._status[2] = (_Status) dp.Get<int>();
	  _transformations[i]._status[3] = (_Status) dp.Get<int>();
	  _transformations[i]._status[4] = (_Status) dp.Get<int>();
	  _transformations[i]._status[5] = (_Status) dp.Get<int>();
	      
	  auto rows = dp.Get<int>();
	  auto cols = dp.Get<int>();
	  
	  dp.Get(rows*cols*sizeof(double), (uint8_t*)_transformations[i]._matrix.GetMatrix());
      }

      InitializeEMValues();
      CoeffInit(1);
      
      // sending
      auto retbuf = MakeUniqueIOBuf(1 * sizeof(int));
      auto retdp = retbuf->GetMutDataPointer();
      retdp.Get<int>() = ret;
      SendMessage(nids[0], std::move(retbuf));
      
  }
  else 
  {
      FORPRINTF("EbbRT BM: ERROR unknown command\n");
  }
#else
  // [SMAFJAS] FRONTEND RECEIVE MESSAGE
  if (ret == 0) 
  {
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 1)
  {
      int start = dp.Get<int>();
      int end = dp.Get<int>();
      
      if(gaussreconptr == NULL) {
	  // malloc using _end and _start since can't guarantee end-start is the largest
	  // buffer possible
	  gaussreconptr = (int*) malloc((_end-_start)*sizeof(int));
      }
      

      dp.Get((end-start)*sizeof(int), (uint8_t*)gaussreconptr);

      // [SMAFJAS] GATHERS ALL DATA 
      memcpy(_voxel_num.data()+start, gaussreconptr, (end-start)*sizeof(int));

      int reconSize = dp.Get<int>();
      if(gaussreconptr2 == NULL)
      {
	  gaussreconptr2 = (double*) malloc (_reconstructed.GetSizeMat()*sizeof(double));
      }
      // [SMAFJAS] EXTRACTING THE POINT WHERE THE DATA IS. COPY THE DATE INTO GAUSSRECONPTR2
      dp.Get(reconSize*sizeof(double), (uint8_t*)gaussreconptr2);
      
      _reconstructed.SumVec(gaussreconptr2);
      
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  gaussianreconFuture.SetValue(1);
      }
  }
  else if(ret == 2)
  {
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 3)
  {
      int num = dp.Get<int>();
      double sigma = dp.Get<double>();
      
      _tsigma += sigma;
      _tnum += num;
      
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 4)
  {
      double tmp;

      tmp = dp.Get<double>();
      _tsum += tmp;

      tmp = dp.Get<double>();
      _tden += tmp;

      tmp = dp.Get<double>();
      _tsum2 += tmp;
      
      tmp = dp.Get<double>();
      _tden2 += tmp;

      tmp = dp.Get<double>();
      if(tmp > _tmaxs) _tmaxs = tmp;
      
      tmp = dp.Get<double>();
      if(tmp < _tmins) _tmins = tmp;

      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if (ret == 5)
  {
      _ttsum += dp.Get<double>();
      _ttden += dp.Get<double>();
      _ttsum2 += dp.Get<double>();
      _ttden2 += dp.Get<double>();
      
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if (ret == 6)
  {
      _ttsum += dp.Get<double>();
      _ttnum += dp.Get<int>();

      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if (ret == 7)
  {
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if (ret == 8)
  {
      int addonSize = dp.Get<int>();
      dp.Get(addonSize*sizeof(double), (uint8_t*)gaussreconptr2);
      _addon.SumVec(gaussreconptr2);

      int confidenceMapSize = dp.Get<int>();
      dp.Get(confidenceMapSize*sizeof(double), (uint8_t*)gaussreconptr2);
      _confidence_map.SumVec(gaussreconptr2);
      
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 10)
  {
      _msigma += dp.Get<double>();
      _mmix += dp.Get<double>();
      _mnum += dp.Get<double>();

      double tmp = dp.Get<double>();
      if(tmp < _mmin) _mmin = tmp;

      tmp = dp.Get<double>();
      if(tmp > _mmax) _mmax = tmp;

      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 11)
  {
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if(ret == 12)
  {
      double ssn = dp.Get<double>();
      double ssd = dp.Get<double>();

      // FORPRINTF("%lf %lf %lf %lf\n", _sscalenum, _sscaleden, ssn, ssd);

      _sscalenum += ssn;
      _sscaleden += ssd;
      
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  gaussianreconFuture.SetValue(1);
      }
  }
  else if(ret == 13)
  {
      int start = dp.Get<int>();
      int end = dp.Get<int>();

      for(int i = start; i < end; i ++)
      {
	  _transformations[i]._tx = dp.Get<double>();
	  _transformations[i]._ty = dp.Get<double>();
	  _transformations[i]._tz = dp.Get<double>();

	  _transformations[i]._rx = dp.Get<double>();
	  _transformations[i]._ry = dp.Get<double>();
	  _transformations[i]._rz = dp.Get<double>();

	  _transformations[i]._cosrx = dp.Get<double>();
	  _transformations[i]._cosry = dp.Get<double>();
	  _transformations[i]._cosrz = dp.Get<double>();

	  _transformations[i]._sinrx = dp.Get<double>();
	  _transformations[i]._sinry = dp.Get<double>();
	  _transformations[i]._sinrz = dp.Get<double>();
	
	  _transformations[i]._status[0] = (_Status) dp.Get<int>();
	  _transformations[i]._status[1] = (_Status) dp.Get<int>();
	  _transformations[i]._status[2] = (_Status) dp.Get<int>();
	  _transformations[i]._status[3] = (_Status) dp.Get<int>();
	  _transformations[i]._status[4] = (_Status) dp.Get<int>();
	  _transformations[i]._status[5] = (_Status) dp.Get<int>();
	      
	  auto rows = dp.Get<int>();
	  auto cols = dp.Get<int>();
	  
	  dp.Get(rows*cols*sizeof(double), (uint8_t*)_transformations[i]._matrix.GetMatrix());
      }

      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else if (ret == 14) 
  {
      reconRecv++;
      if (reconRecv == numNodes) 
      {
	  reconRecv = 0;
	  testFuture.SetValue(1);
      }
  }
  else 
  {
    FORPRINTF("EbbRT HOSTED: ERROR unknown command\n");
  }

#endif
}

#pragma GCC diagnostic pop

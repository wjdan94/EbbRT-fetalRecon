#include <irtkImage.h>
#include <irtkTransformation.h>

#include <ebbrt/IOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticIOBuf.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

using namespace std;
using namespace ebbrt;

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImage(
    irtkRealImage& img);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageAttr(
    irtkRealImage ri);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageI2W(
    irtkRealImage& ri);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageW2I(
    irtkRealImage& ri);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeSlice(
    irtkRealImage& ri);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeRigidTrans(
    irtkRigidTransformation& rt);

inline void deserializeSlice(ebbrt::IOBuf::DataPointer& dp, 
    irtkRealImage& tmp);

inline void deserializeTransformations(
    ebbrt::IOBuf::DataPointer& dp, irtkRigidTransformation& tmp);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeTransformations(
    vector<irtkRigidTransformation>& transformations);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeSlices(
    vector<irtkRealImage>& slices);

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeTransformations(
    vector<irtkRigidTransformation>& transformations) {
  auto buf = MakeUniqueIOBuf(1 * sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = transformations.size();

  for(int j = 0; j < transformations.size(); j++) {
    buf->PrependChain(std::move(serializeRigidTrans(transformations[j])));
  }

  return buf;
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageAttr(irtkRealImage ri) {
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

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageI2W(
    irtkRealImage& ri) {
  auto buf = MakeUniqueIOBuf(2 * sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Rows();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Cols();

  auto buf2 = std::make_unique<StaticIOBuf>(
      reinterpret_cast<const uint8_t *>(ri.GetWorldToImageMatrix().GetMatrix()),
      (size_t)(ri.GetWorldToImageMatrix().Rows() * 
        ri.GetWorldToImageMatrix().Cols() * sizeof(double)));
  buf->PrependChain(std::move(buf2));

  return buf;
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageW2I(
    irtkRealImage& ri) {
  auto buf = MakeUniqueIOBuf(2 * sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Rows();
  dp.Get<int>() = ri.GetWorldToImageMatrix().Cols();

  auto buf2 = std::make_unique<StaticIOBuf>(
      reinterpret_cast<const uint8_t *>(ri.GetWorldToImageMatrix().GetMatrix()),
      (size_t)(ri.GetWorldToImageMatrix().Rows() * 
        ri.GetWorldToImageMatrix().Cols() * sizeof(double)));
  buf->PrependChain(std::move(buf2));

  return buf;
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeSlice(
    irtkRealImage& ri) {
  auto buf = MakeUniqueIOBuf(sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = ri.GetSizeMat();

  auto buf2 = std::make_unique<StaticIOBuf>(
      reinterpret_cast<const uint8_t *>(ri.GetMat()),
      (size_t)(ri.GetSizeMat() * sizeof(double)));

  buf->PrependChain(std::move(buf2));

  return buf;
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImage(
    irtkRealImage& img) {
  auto buf = MakeUniqueIOBuf(0);
  buf->PrependChain(std::move(serializeImageAttr(img)));
  buf->PrependChain(std::move(serializeImageI2W(img)));
  buf->PrependChain(std::move(serializeImageW2I(img)));
  buf->PrependChain(std::move(serializeSlice(img)));
  return buf;
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeRigidTrans(
    irtkRigidTransformation& rt) {
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

inline void deserializeSlice(ebbrt::IOBuf::DataPointer& dp, 
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
  auto ptr = std::make_unique<double[]>(rows * cols);
  dp.Get(rows * cols * sizeof(double), (uint8_t*)ptr.get());
  irtkMatrix matI2W(rows, cols, std::move(ptr));

  rows = dp.Get<int>();
  cols = dp.Get<int>();
  ptr = std::make_unique<double[]>(rows * cols);
  dp.Get(rows * cols * sizeof(double), (uint8_t*)ptr.get());
  irtkMatrix matW2I(rows, cols, std::move(ptr));

  auto n = dp.Get<int>();
  auto ptr2 = new double[n];
  dp.Get(n*sizeof(double), (uint8_t*)ptr2);

  irtkRealImage ri(at, ptr2, matI2W, matW2I);

  tmp = std::move(ri);
}

inline void deserializeTransformations(
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

  irtkRigidTransformation irt(tx, ty, tz, rx, ry, rz, cosrx, cosry,cosrz, sinx,
      siny, sinz, status0, status1, status2, status3, status4, status5, mat);

  tmp = std::move(irt);
}

inline std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeSlices(
    vector<irtkRealImage>& slices) {
  auto buf = MakeUniqueIOBuf(sizeof(int));
  auto dp = buf->GetMutDataPointer();
  dp.Get<int>() = slices.size();

  for (int j = 0; j < slices.size(); j++) {
    buf->PrependChain(std::move(serializeImageAttr(slices[j])));
    buf->PrependChain(std::move(serializeImageI2W(slices[j])));
    buf->PrependChain(std::move(serializeImageW2I(slices[j])));
    buf->PrependChain(std::move(serializeSlice(slices[j])));
  }

  return buf;
}

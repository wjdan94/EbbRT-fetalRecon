project("ebbrt-fetal-reconstruction" C CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")

set(IRTK_SOURCE_DIR IRTKSimple2/)

set(IRTK_IMAGE_SRC
  IRTKSimple2/image++/src/irtkGenericImage.cc
  IRTKSimple2/image++/src/irtkBaseImage.cc
  IRTKSimple2/image++/src/irtkImageToFile.cc
  IRTKSimple2/image++/src/irtkImageToFileGIPL.cc
  IRTKSimple2/image++/src/irtkImageToFileVTK.cc
  IRTKSimple2/image++/src/irtkImageToFilePGM.cc
  IRTKSimple2/image++/src/irtkImageToFileANALYZE.cc
  IRTKSimple2/image++/src/irtkImageToFileNIFTI.cc
  IRTKSimple2/image++/src/irtkFileToImage.cc
  IRTKSimple2/image++/src/irtkFileGIPLToImage.cc
  IRTKSimple2/image++/src/irtkFileNIFTIToImage.cc
  IRTKSimple2/image++/src/irtkFileANALYZEToImage.cc
  IRTKSimple2/image++/src/irtkFileVTKToImage.cc
  IRTKSimple2/image++/src/irtkFilePGMToImage.cc 
  IRTKSimple2/image++/src/irtkHistogram_2D.cc 
  IRTKSimple2/image++/src/irtkHistogram_1D.cc 
  IRTKSimple2/image++/src/irtkResamplingWithPadding.cc
  IRTKSimple2/image++/src/irtkNearestNeighborInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkGaussianBlurring.cc
  IRTKSimple2/image++/src/irtkResampling.cc
  IRTKSimple2/image++/src/irtkImageToImage.cc
  IRTKSimple2/image++/src/irtkInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkLinearInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkGaussianBlurringWithPadding.cc
  IRTKSimple2/image++/src/irtkScalarFunctionToImage.cc
  IRTKSimple2/image++/src/irtkConvolution.cc
  IRTKSimple2/image++/src/irtkImageFunction.cc
  IRTKSimple2/image++/src/irtkConvolution_1D.cc
  IRTKSimple2/image++/src/irtkSincInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkGaussianInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkBSplineInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkCSplineInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkShapeBasedInterpolateImageFunction.cc
  IRTKSimple2/image++/src/irtkSincInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/irtkNearestNeighborInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/irtkGaussianInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/irtkBSplineInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/irtkCSplineInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/irtkConvolutionWithPadding_1D.cc
  IRTKSimple2/image++/src/irtkLinearInterpolateImageFunction2D.cc
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
  IRTKSimple2/image++/src/
)

set(IRTK_COMMON_SRC
  IRTKSimple2/common++/src/irtkObject.cc
  IRTKSimple2/common++/src/irtkCifstream.cc
  IRTKSimple2/common++/src/irtkCofstream.cc
  IRTKSimple2/common++/src/swap.cc
  IRTKSimple2/common++/src/basename.cc 
)

set(IRTK_CONTRIB_SRC IRTKSimple2/contrib++/src)

set(IRTK_GEOMETRY_SRC
  IRTKSimple2/geometry++/src/irtkMatrix.cc
  IRTKSimple2/geometry++/src/irtkPointSet.cc
  IRTKSimple2/geometry++/src/irtkVector.cc 
  IRTKSimple2/geometry++/src/irtkScalarGaussian.cc
  IRTKSimple2/geometry++/src/irtkScalarFunction.cc
)

set(IRTK_REGISTRATION_SRC
  IRTKSimple2/packages/registration/src/irtkImageRigidRegistrationWithPadding.cc
  IRTKSimple2/packages/registration/src/irtkImageRegistration.cc
  IRTKSimple2/packages/registration/src/irtkImageRegistrationWithPadding.cc
  IRTKSimple2/packages/registration/src/irtkUtil.cc
  IRTKSimple2/packages/registration/src/irtkOptimizer.cc
  IRTKSimple2/packages/registration/src/irtkSteepestGradientDescentOptimizer.cc
  IRTKSimple2/packages/registration/src/irtkGradientDescentOptimizer.cc
  IRTKSimple2/packages/registration/src/irtkDownhillDescentOptimizer.cc
  IRTKSimple2/packages/registration/src/irtkConjugateGradientDescentOptimizer.cc
  IRTKSimple2/packages/registration/src/irtkGradientDescentConstrainedOptimizer.cc
)

set(IRTK_TRANSFORMATION_SRC 
  IRTKSimple2/packages/transformation/src/irtkTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkTemporalHomogeneousTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkRigidTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkAffineTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkTemporalAffineTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkHomogeneousTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkTemporalRigidTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkImageTransformation.cc
  IRTKSimple2/packages/transformation/src/irtkImageHomogeneousTransformation.cc
  IRTKSimple2/packages/transformation/src/
  IRTKSimple2/packages/transformation/src/
  IRTKSimple2/packages/transformation/src/
  IRTKSimple2/packages/transformation/src/
)

set(IRTK_NIFTI_SRC
  IRTKSimple2/nifti/niftilib/nifti1_io.c
)

set(IRTK_ZNZ_SRC
  IRTKSimple2/nifti/znzlib/znzlib.c
)

set(BAREMETAL_SOURCES
  src/baremetal/irtkReconstruction.cc
)

# set(BAREMETAL_INCLUDES
#  src/baremetal
#
# ) 

set(HOSTED_SOURCES
  src/hosted/reconstruction.cc
  src/hosted/irtkReconstruction.cc
)

set(EXT
  ext/
)

set(EXT_LIBS

)
# Baremetal  ========================================================

if( ${CMAKE_SYSTEM_NAME} STREQUAL "EbbRT")

  message(STATUS "### BUILDING NATIVE ###")
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror -Wno-unused-local-typedefs -Wno-unused-label -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable -Wno-deprecated -DHAS_NIFTI -DHAVE_INLINE -D__MNODE__ -D__EBBRT_BM__")

  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/common++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/geometry++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/image++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/contrib++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/packages/transformation/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/packages/registration/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/nifti/niftilib)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/nifti/znzlib)
  INCLUDE_DIRECTORIES(${EXT})
  
  include_directories(${BAREMETAL_INCLUDES})

  add_executable(reconstruction.elf ${BAREMETAL_SOURCES} ${IRTK_IMAGE_SRC} ${IRTK_COMMON_SRC} ${IRTK_GEOMETRY_SRC} ${IRTK_contrib_SRC} ${IRTK_NIFTI_SRC} ${IRTK_ZNZ_SRC} ${IRTK_TRANSFORMATION_SRC} ${IRTK_REGISTRATION_SRC})

  target_link_libraries(reconstruction.elf ${CMAKE_SOURCE_DIR}/libs/libboost_serialization.a ${CMAKE_SOURCE_DIR}/libs/libboost_system.a ${CMAKE_SOURCE_DIR}/libs/libboost_wserialization.a ${CMAKE_SOURCE_DIR}/libs/libgsl.a ${CMAKE_SOURCE_DIR}/libs/libgslcblas.a)
  
  add_custom_command(TARGET reconstruction.elf POST_BUILD 
    COMMAND objcopy -O elf32-i386 reconstruction.elf reconstruction.elf32 )
  
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
  message(STATUS "########################### BUILDING HOSTED LINUX #########################")

  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror -D__EBB__ -DHAS_NIFTI -DHAS_TBB -DHAVE_INLINE -D__MNODE__ -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-label -Wno-unused-function")

  find_package(EbbRT REQUIRED)
  find_package(Boost 1.53.0 REQUIRED COMPONENTS
    filesystem system coroutine context program_options)
  find_package(Capnp REQUIRED)
  find_package(TBB REQUIRED)
  find_package(Threads REQUIRED)
 
  FIND_PACKAGE(GSL REQUIRED)
  include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})

  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/common++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/geometry++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/image++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/contrib++/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/packages/transformation/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/packages/registration/include)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/nifti/niftilib)
  INCLUDE_DIRECTORIES(${IRTK_SOURCE_DIR}/nifti/znzlib)
  
  include_directories(${EBBRT_INCLUDE_DIRS})
  add_executable(reconstruction ${HOSTED_SOURCES} ${IRTK_IMAGE_SRC} ${IRTK_COMMON_SRC} ${IRTK_GEOMETRY_SRC} ${IRTK_contrib_SRC} ${IRTK_NIFTI_SRC} ${IRTK_ZNZ_SRC} ${IRTK_TRANSFORMATION_SRC} ${IRTK_REGISTRATION_SRC})
  
  target_link_libraries(reconstruction
    ${EBBRT_LIBRARIES} ${CAPNP_LIBRARIES_LITE} ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${GSL_LIBRARIES}
  )
  
else()
  message(FATAL_ERROR "System name unsupported: ${CMAKE_SYSTEM_NAME}")
endif()

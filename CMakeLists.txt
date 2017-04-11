project("ebbrt-fetal-reconstruction" C CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(IRTK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/fetalReconstruction/source/IRTKSimple2")
set(GSL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/EbbRT-libgsl")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror -DHAS_NIFTI -DHAVE_INLINE -D__MNODE__ -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-label -Wno-unused-function -Wno-deprecated -Wno-unused-local-typedefs -Wno-unused-variable")

# IRTK
set(IRTK_SUBDIRS
  ${IRTK_SOURCE_DIR}/common++/src 
  ${IRTK_SOURCE_DIR}/geometry++/src 
  ${IRTK_SOURCE_DIR}/image++/src 
  ${IRTK_SOURCE_DIR}/contrib++/src 
  ${IRTK_SOURCE_DIR}/packages 
  ${IRTK_SOURCE_DIR}/nifti/znzlib 
  ${IRTK_SOURCE_DIR}/nifti/niftilib
)
set(IRTK_INCLUDE_DIRS
  ${IRTK_SOURCE_DIR}/common++/include
  ${IRTK_SOURCE_DIR}/geometry++/include
  ${IRTK_SOURCE_DIR}/image++/include
  ${IRTK_SOURCE_DIR}/contrib++/include
  ${IRTK_SOURCE_DIR}/packages/transformation/include
  ${IRTK_SOURCE_DIR}/packages/registration/include
  ${IRTK_SOURCE_DIR}/nifti/niftilib
  ${IRTK_SOURCE_DIR}/nifti/znzlib
)
  
if( ${CMAKE_SYSTEM_NAME} STREQUAL "EbbRT")
  # Native Build  ========================================================
  # Build GSL library from source
  subdirs(${GSL_SOURCE_DIR}) 
  include_directories(${GSL_SOURCE_DIR}/include)
  # IRTK
  subdirs(${IRTK_SUBDIRS}) 
  include_directories(${IRTK_INCLUDE_DIRS})
  # App target 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__EBBRT_BM__")
  add_executable(reconstruction.elf src/baremetal/irtkReconstruction.cc )
  target_link_libraries(reconstruction.elf registration++ transformation++
    contrib++ image++ geometry++ common++ niftiio znz gsl ) 
  add_custom_command(TARGET reconstruction.elf POST_BUILD 
    COMMAND objcopy -O elf32-i386 reconstruction.elf reconstruction.elf32 )
  
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
  # Hosted Build  ========================================================
  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_TBB ")
  find_package(EbbRT REQUIRED)
  find_package(Boost 1.53.0 REQUIRED COMPONENTS
    filesystem system coroutine context program_options)
  find_package(Capnp REQUIRED)
  find_package(TBB REQUIRED)
  find_package(Threads REQUIRED)
  find_package(GSL REQUIRED)
  # GSL
  include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
  # IRTK
  subdirs(${IRTK_SUBDIRS}) 
  include_directories(${IRTK_INCLUDE_DIRS})
  include_directories(${EBBRT_INCLUDE_DIRS})
  add_executable(reconstruction src/hosted/reconstruction.cc src/hosted/irtkReconstruction.cc)
  target_link_libraries(reconstruction registration++ transformation++
    contrib++ image++ geometry++ common++ niftiio znz ${CMAKE_THREAD_LIBS_INIT}
    ${EBBRT_LIBRARIES} ${CAPNP_LIBRARIES_LITE} 
    ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${GSL_LIBRARIES}
    )
else()
  message(FATAL_ERROR "System name unsupported: ${CMAKE_SYSTEM_NAME}")
endif()

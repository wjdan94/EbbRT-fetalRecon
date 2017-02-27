//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#define NOMINMAX
#define _USE_MATH_DEFINES

#include <math.h>
#include <stdlib.h>
//#include "../../src/utils.h"
#include <chrono>
#include <signal.h>
#include <string>
#include <thread>

#include "irtkReconstructionEbb.h"

#if 1
#include <irtkRegistration.h> //this header needs to be at top else compilation error
#include <irtkImageFunction.h>
#endif

#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkResampling.h>
#include <irtkTransformation.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <ebbrt/Cpu.h>

#include <ebbrt/hosted/Clock.h>
#include <ebbrt/hosted/Context.h>
#include <ebbrt/hosted/ContextActivation.h>
#include <ebbrt/hosted/GlobalIdMap.h>
#include <ebbrt/hosted/NodeAllocator.h>
#include <ebbrt/hosted/PoolAllocator.h>

#include <ebbrt/hosted/Runtime.h>
#include <ebbrt/hosted/StaticIds.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

// TODO: fix this later
int ARGC;
char** ARGV;

namespace po = boost::program_options;

using namespace ebbrt;

float msumImage(std::vector<irtkRealImage> a)
{
    float sum = 0.0;
    for (unsigned int i = 0; i < a.size(); i++) 
    {
	irtkRealPixel *ap = a[i].GetPointerToVoxels();
	for(int j = 0; j < (int)a[i].GetNumberOfVoxels(); j++)
	{
	    sum += *ap;
	    ap ++;
	}
    }
    return sum;
}

const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

  return buf;
}

void AppMain() {
  struct timeval totstart, totend;
  gettimeofday(&totstart, NULL);
  int i, ok;
  char buffer[256];
  irtkRealImage stack;

  // declare variables for input
  /// Slice stacks
  vector<irtkRealImage> stacks;
  /// Stack transformation
  vector<irtkRigidTransformation> stack_transformations;
  /// Stack thickness
  vector<double> thickness;
  /// number of stacks
  int nStacks;
  /// number of packages for each stack
  vector<int> packages;

  vector<float> stackMotion;

  // Default values.
  int templateNumber = -1;
  irtkRealImage *mask = NULL;
  int iterations = 9; // 9 //2 for Shepp-Logan is enough
  bool debug = false;
  bool debug_gpu = false;
  double sigma = 20;
  double resolution = 0.75;
  double lambda = 0.02;
  double delta = 150;
  int levels = 3;
  double lastIterLambda = 0.01;
  int rec_iterations;
  double averageValue = 700;
  double smooth_mask = 4;
  bool global_bias_correction = false;
  double low_intensity_cutoff = 0.01;
  // folder for slice-to-volume registrations, if given
  string tfolder;
  // folder to replace slices with registered slices, if given
  string sfolder;
  // flag to swich the intensity matching on and off
  bool intensity_matching = true;
  int rec_iterations_first = 4;
  int rec_iterations_last = 13;

  // number of threads
  int numThreads;
  int numNodes;

  bool useCPU = false;
  bool useCPUReg = true;
  bool useGPUReg = false;
  bool disableBiasCorr = false;
  bool useAutoTemplate = false;

  irtkRealImage average;

  string log_id;
  bool no_log = false;

  // forced exclusion of slices
  int number_of_force_excluded_slices = 0;
  vector<int> force_excluded;
  vector<int> devicesToUse;

  vector<string> inputStacks;
  vector<string> inputTransformations;
  string maskName;
  /// Name for output volume
  string outputName;
  unsigned int num_input_stacks_tuner = 0;
  
  string referenceVolumeName;
  unsigned int T1PackageSize = 0;
  unsigned int numDevicesToUse = UINT_MAX;
  bool useSINCPSF = false;
  bool serial = false;
  
  try {
    po::options_description desc("Options");
    desc.add_options()("help,h", "Print usage messages")(
        "output,o", po::value<string>(&outputName)->required(),
        "Name for the reconstructed volume. Nifti or Analyze format.")(
        "mask,m", po::value<string>(&maskName), "Binary mask to define the "
                                                "region od interest. Nifti or "
                                                "Analyze format.")(
        "input,i", po::value<vector<string>>(&inputStacks)->multitoken(),
        "[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format.")(
        "transformation,t",
        po::value<vector<string>>(&inputTransformations)->multitoken(),
        "The transformations of the input stack to template in \'dof\' format "
        "used in IRTK. Only rough alignment with correct orienation and some "
        "overlap is needed. Use \'id\' for an identity transformation for at "
        "least one stack. The first stack with \'id\' transformation  will be "
        "resampled as template.")(
        "thickness", po::value<vector<double>>(&thickness)->multitoken(),
        "[th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z "
        "direction]")(
        "packages,p", po::value<vector<int>>(&packages)->multitoken(),
        "Give number of packages used during acquisition for each stack. The "
        "stacks will be split into packages during registration iteration 1 "
        "and then into odd and even slices within each package during "
        "registration iteration 2. The method will then continue with slice to "
        " volume approach. [Default: slice to volume registration only]")(
        "iterations", po::value<int>(&iterations)->default_value(1),
        "Number of registration-reconstruction iterations.")(
        "sigma", po::value<double>(&sigma)->default_value(12.0),
        "Stdev for bias field. [Default: 12mm]")(
        "resolution", po::value<double>(&resolution)->default_value(0.75),
        "Isotropic resolution of the volume. [Default: 0.75mm]")(
        "multires", po::value<int>(&levels)->default_value(3),
        "Multiresolution smooting with given number of levels. [Default: 3]")(
        "average", po::value<double>(&averageValue)->default_value(700),
        "Average intensity value for stacks [Default: 700]")(
        "delta", po::value<double>(&delta)->default_value(150),
        " Parameter to define what is an edge. [Default: 150]")(
        "lambda", po::value<double>(&lambda)->default_value(0.02),
        "  Smoothing parameter. [Default: 0.02]")(
        "lastIterLambda",
        po::value<double>(&lastIterLambda)->default_value(0.01),
        "Smoothing parameter for last iteration. [Default: 0.01]")(
        "smooth_mask", po::value<double>(&smooth_mask)->default_value(4),
        "Smooth the mask to reduce artefacts of manual segmentation. [Default: "
        "4mm]")(
        "global_bias_correction",
        po::value<bool>(&global_bias_correction)->default_value(false),
        "Correct the bias in reconstructed image against previous estimation.")(
        "low_intensity_cutoff",
        po::value<double>(&low_intensity_cutoff)->default_value(0.01),
        "Lower intensity threshold for inclusion of voxels in global bias "
        "correction.")("force_exclude",
                       po::value<vector<int>>(&force_excluded)->multitoken(),
                       "force_exclude [number of slices] [ind1] ... [indN]  "
                       "Force exclusion of slices with these indices.")(
        "no_intensity_matching", po::value<bool>(&intensity_matching),
        "Switch off intensity matching.")(
        "log_prefix", po::value<string>(&log_id), "Prefix for the log file.")(
        "debug", po::value<bool>(&debug)->default_value(false),
        " Debug mode - save intermediate results.")(
        "debug_gpu", po::bool_switch(&debug_gpu)->default_value(false),
        " Debug only GPU results.")(
        "rec_iterations_first",
        po::value<int>(&rec_iterations_first)->default_value(4),
        " Set number of superresolution iterations")(
        "rec_iterations_last",
        po::value<int>(&rec_iterations_last)->default_value(13),
        " Set number of superresolution iterations for the last iteration")(
        "num_stacks_tuner",
        po::value<unsigned int>(&num_input_stacks_tuner)->default_value(0),
        "  Set number of input stacks that are really used (for tuner "
        "evaluation, use only first x)")(
        "no_log", po::value<bool>(&no_log)->default_value(false),
        "  Do not redirect cout and cerr to log files.")(
        "devices,d", po::value<vector<int>>(&devicesToUse)->multitoken(),
        "  Select the CP > 3.0 GPUs on which the reconstruction should be "
        "executed. Default: all devices > CP 3.0")(
        "tfolder", po::value<string>(&tfolder),
        "[folder] Use existing slice-to-volume transformations to initialize "
        "the reconstruction.")("sfolder", po::value<string>(&sfolder),
                               "[folder] Use existing registered slices and "
                               "replace loaded ones (have to be equally many "
                               "as loaded from stacks).")(
        "referenceVolume", po::value<string>(&referenceVolumeName),
        "Name for an optional reference volume. Will be used as inital "
        "reconstruction.")("T1PackageSize",
                           po::value<unsigned int>(&T1PackageSize),
                           "is a test if you can register T1 to T2 using NMI "
                           "and only one iteration")(
        "numDevicesToUse", po::value<unsigned int>(&numDevicesToUse),
        "sets how many GPU devices to use in case of automatic device "
        "selection. Default is as many as available.")(
        "useCPU", po::bool_switch(&useCPU)->default_value(false),
        "use CPU for reconstruction and registration; performs superresolution "
        "and robust statistics on CPU. Default is using the GPU")(
        "useCPUReg", po::bool_switch(&useCPUReg)->default_value(true),
        "use CPU for more flexible CPU registration; performs superresolution "
        "and robust statistics on GPU. [default, best result]")(
        "useGPUReg", po::bool_switch(&useGPUReg)->default_value(false),
        "use faster but less accurate and flexible GPU registration; performs "
        "superresolution and robust statistics on GPU.")(
        "useAutoTemplate",
        po::bool_switch(&useAutoTemplate)->default_value(false),
        "select 3D registration template stack automatically with matrix rank "
        "method.")("useSINCPSF",
                   po::bool_switch(&useSINCPSF)->default_value(false),
                   "use a more MRI like SINC point spread function (PSF) Will "
                   "be in plane sinc (Bartlett) and through plane Gaussian.")(
        "disableBiasCorrection",
        po::bool_switch(&disableBiasCorr)->default_value(false),
        "disable bias field correction for cases with little or no bias field "
        "inhomogenities (makes it faster but less reliable for stron intensity "
        "bias)")
	("numThreads", po::value<int>(&numThreads)->default_value(1),
                 "Number of CPU threads to run for TBB")("numNodes", po::value<int>(&numNodes)->default_value(1),
                 "Number of back-end EbbRT nodes");
    
    po::variables_map vm;

    try {
      po::store(po::command_line_parser(ARGC, ARGV).options(desc).allow_unregistered().run(), vm);

      if (vm.count("help")) {
        std::cout << "Application to perform reconstruction of volumetric MRI "
                     "from thick slices."
                  << std::endl
                  << desc << std::endl;
      }

      po::notify(vm);
    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      exit(EXIT_FAILURE);
    }
  } catch (std::exception &e) {
    std::cerr << "Unhandled exception while parsing arguments:  " << e.what()
              << ", application will now exit" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (useCPU) {
    // security measure for wrong input params
    useCPUReg = true;
    useGPUReg = false;

    // set CPU  threads
    if (numThreads > 0) {
      //      cout << "numThreads = " << numThreads << endl;
    } else {
      //    cout << "Using task_scheduler_init::automatic number of threads" <<
      //    endl;
    }
  }

  //  cout << "Reconstructed volume name ... " << outputName << endl;
  nStacks = inputStacks.size();
  // cout << "Number of stacks ... " << nStacks << endl;

  float tmp_motionestimate = FLT_MAX;
  for (i = 0; i < nStacks; i++) {
    stack.Read(inputStacks[i].c_str());
    // cout << "Reading stack ... " << inputStacks[i] << endl;
    stacks.push_back(stack);

    //cout << i << " " << stack.Sum() << endl;
  }
  //cout << "1 stacks[0] =  " << stacks[0].Sum() << endl;

  for (i = 0; i < nStacks; i++) {
    irtkTransformation *transformation;
    if (!inputTransformations.empty()) {
      try {
        transformation =
            irtkTransformation::New((char *)(inputTransformations[i].c_str()));
      } catch (...) {
        transformation = new irtkRigidTransformation;
        if (templateNumber < 0)
          templateNumber = 0;
      }
    } else {
      transformation = new irtkRigidTransformation;
      if (templateNumber < 0)
        templateNumber = 0;
    }

    irtkRigidTransformation *rigidTransf =
        dynamic_cast<irtkRigidTransformation *>(transformation);
    stack_transformations.push_back(*rigidTransf);
    delete rigidTransf;
  }

  //cout << "2 stacks[0] =  " << stacks[0].Sum() << endl;
  // std::printf("*********** Create() ***********\n");

  auto reconstruction = irtkReconstructionEbb::Create();
  reconstruction->setNumThreads(numThreads);
  reconstruction->setNumNodes(numNodes);


  auto bindir = boost::filesystem::system_complete(ARGV[0]).parent_path() /
                "/bm/reconstruction.elf32";

  struct timeval allocation_time_start;

  gettimeofday(&allocation_time_start, NULL);

  try {
    ebbrt::pool_allocator->AllocatePool(bindir.string(), numNodes);
  } catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Initialize Pool futures
  for (unsigned int i=0 ; i < numNodes; i++) {
    pool_allocator->pool_futures.push_back(Promise<int>());
  }
  
  for (unsigned int i=0 ; i < numNodes; i++) {
    // Waiting for Future
    auto nf = pool_allocator->pool_futures[i].GetFuture();
    nf.Block();
    pool_allocator->node_descriptors_[i].NetworkId().Then([reconstruction, i, numNodes, allocation_time_start](Future<Messenger::NetworkId> f) {
        auto nid = f.Get();
        reconstruction->addNid(nid);
    });
  }

  if (useSINCPSF) {
    reconstruction->useSINCPSF();
  }

  reconstruction->InvertStackTransformations(stack_transformations);

  if (!maskName.empty()) {
    mask = new irtkRealImage((char *)(maskName.c_str()));
  }

  if (num_input_stacks_tuner > 0) {
    nStacks = num_input_stacks_tuner;
    //    cout << "actually used stacks for tuner test .... "
    //       << num_input_stacks_tuner << endl;
  }

  number_of_force_excluded_slices = force_excluded.size();

  //cout << "3 stacks[0] =  " << stacks[0].Sum() << endl;
  // erase stacks for tuner evaluation
  if (num_input_stacks_tuner > 0) {
    stacks.erase(stacks.begin() + num_input_stacks_tuner, stacks.end());
    stack_transformations.erase(stack_transformations.begin() +
                                    num_input_stacks_tuner,
                                stack_transformations.end());
    //    std::cout << "stack sizes: " << nStacks << " " << stacks.size() << " "
    //            << thickness.size() << " " << stack_transformations.size()
    //        << std::endl;
  }

  //cout << "4 stacks[0] =  " << stacks[0].Sum() << endl;

  // Initialise 2*slice thickness if not given by user
  if (thickness.size() == 0) {
    //    cout << "Slice thickness is ";
    for (i = 0; i < nStacks; i++) {
      double dx, dy, dz;
      stacks[i].GetPixelSize(&dx, &dy, &dz);
      thickness.push_back(dz * 2);
      //    cout << thickness[i] << " ";
    }
    // cout << "." << endl;
  }

  //cout << "5 stacks[0] =  " << stacks[0].Sum() << endl;
  // Output volume
  irtkRealImage reconstructed;
  irtkRealImage lastReconstructed;
  irtkRealImage reconstructedGPU;

  std::vector<double> samplingUcert;

  // Set debug mode
  if (debug)
    reconstruction->DebugOn();
  else
    reconstruction->DebugOff();

  // Set force excluded slices
  reconstruction->SetForceExcludedSlices(force_excluded);

  // Set low intensity cutoff for bias estimation
  reconstruction->SetLowIntensityCutoff(low_intensity_cutoff);

  // Check whether the template stack can be indentified
  if (templateNumber < 0) {
    cerr << "Please identify the template by assigning id transformation."
         << endl;
    exit(1);
  }
  // If no mask was given  try to create mask from the template image in case it
  // was padded
  if ((mask == NULL) && (sfolder.empty())) {
    mask = new irtkRealImage(stacks[templateNumber]);
    *mask = reconstruction->CreateMask(*mask);
  }

  //cout << "6 stacks[0] =  " << stacks[0].Sum() << endl;
  
  // copy to tmp stacks for template determination
  std::vector<irtkRealImage> tmpStacks;
  for (i = 0; i < stacks.size(); i++) {
    tmpStacks.push_back(stacks[i]);
  }

  // Before creating the template we will crop template stack according to the
  // given mask
  if (mask != NULL) {
    // first resample the mask to the space of the stack
    // for template stact the transformation is identity
    irtkRealImage m = *mask;

    // now do it really with best stack
    reconstruction->TransformMask(stacks[templateNumber], m,
                                  stack_transformations[templateNumber]);
    //cout << "1 stacks[0] =  " << stacks[0].Sum() << endl;

    // Crop template stack
    reconstruction->CropImage(stacks[templateNumber], m);

    //cout << "2 stacks[0] =  " << stacks[0].Sum() << endl;

    if (debug) {
      m.Write("maskTemplate.nii.gz");
      stacks[templateNumber].Write("croppedTemplate.nii.gz");
    }
  }

  tmpStacks.erase(tmpStacks.begin(), tmpStacks.end());

  //cout << "7 stacks[0] =  " << stacks[0].Sum() << endl;
  
  std::vector<uint3> stack_sizes;
  uint3 temp; // = (uint3) malloc(sizeof(uint3));
  for (int i = 0; i < stacks.size(); i++) {
    temp.x = stacks[i].GetX();
    temp.y = stacks[i].GetY();
    temp.z = stacks[i].GetZ();
    stack_sizes.push_back(temp);
  }

  // Create template volume with isotropic resolution
  // if resolution==0 it will be determined from in-plane resolution of the
  // image
  resolution =
      reconstruction->CreateTemplate(stacks[templateNumber], resolution);

  // Set mask to reconstruction object.
  reconstruction->SetMask(mask, smooth_mask);

  //cout << "8 stacks[0] =  " << stacks[0].Sum() << endl;

  // to redirect output from screen to text files
  if (T1PackageSize == 0 && sfolder.empty()) {
      //std::cout << "StackRegistrations start" << std::endl;
    // volumetric registration
    reconstruction->StackRegistrations(stacks, stack_transformations,
                                       templateNumber);
  }

  //  std::cout << "reconstruction->CreateAverage" << std::endl;
  average = reconstruction->CreateAverage(stacks, stack_transformations);

  // Mask is transformed to the all other stacks and they are cropped
  for (i = 0; i < nStacks; i++) {
    // template stack has been cropped already
    if ((i == templateNumber))
      continue;

    
    // transform the mask
    irtkRealImage m = reconstruction->GetMask();
    reconstruction->TransformMask(stacks[i], m, stack_transformations[i]);

    // Crop template stack
    reconstruction->CropImage(stacks[i], m);
  }

  if (T1PackageSize == 0 && sfolder.empty()) {
    // volumetric registration
    reconstruction->StackRegistrations(stacks, stack_transformations,
                                       templateNumber);
    //    cout << endl;
  }

  // Rescale intensities of the stacks to have the same average
  if (intensity_matching)
    reconstruction->MatchStackIntensitiesWithMasking(
        stacks, stack_transformations, averageValue);
  else
    reconstruction->MatchStackIntensitiesWithMasking(
        stacks, stack_transformations, averageValue, true);
  average = reconstruction->CreateAverage(stacks, stack_transformations);

  // Create slices and slice-dependent transformations
  // resolution =
  // reconstruction->CreateTemplate(stacks[templateNumber],resolution);
  reconstruction->CreateSlicesAndTransformations(stacks, stack_transformations,
                                                 thickness);

  // Mask all the slices
  reconstruction->MaskSlices();

  // Set sigma for the bias field smoothing
  if (sigma > 0)
    reconstruction->SetSigma(sigma);
  else {
    // cerr<<"Please set sigma larger than zero. Current value: "<<sigma<<endl;
    // exit(1);
    reconstruction->SetSigma(20);
  }

  // Set global bias correction flag
  if (global_bias_correction)
    reconstruction->GlobalBiasCorrectionOn();
  else
    reconstruction->GlobalBiasCorrectionOff();

  // if given read slice-to-volume registrations
  if (!tfolder.empty())
    reconstruction->ReadTransformation((char *)tfolder.c_str());

  //std::printf("$$$$ main_1 _max_intensity = %lf, _min_intensity = %lf, _slices = %f\n\n", reconstruction->_max_intensity, reconstruction->_min_intensity, msumImage(reconstruction->_slices));
  // Initialise data structures for EM
  reconstruction->InitializeEM();

  //std::printf("$$$$ main_2 _max_intensity = %lf, _min_intensity = %lf, _slices = %f\n\n", reconstruction->_max_intensity, reconstruction->_min_intensity, msumImage(reconstruction->_slices));

//  std::cout << "*************** packages.size() " << packages.size()
//          << std::endl;

//  std::printf("lambda = %f delta = %f intensity_matching = %d useCPU = %d
//  disableBiasCorr = %d sigma = %f global_bias_correction = %d lastIterLambda =
//  %f iterations = %d levels = %d\n", lambda, delta, intensity_matching,
//  useCPU, disableBiasCorr, sigma, global_bias_correction, lastIterLambda,
//  iterations, levels);


  reconstruction->waitNodes().Then(
      [reconstruction, iterations](ebbrt::Future<void> f) {
        f.Get();
        std::cout << "All nodes initialized" << std::endl;
        ebbrt::event_manager->Spawn([reconstruction, iterations]() {
          reconstruction->SendRecon(iterations);
        });
      });

  reconstruction->waitReceive().Then([totstart](ebbrt::Future<void> f) {
    f.Get();
    struct timeval totend;
    gettimeofday(&totend, NULL);
    std::printf("tTotal time: %lf seconds\n",
      (totend.tv_sec - totstart.tv_sec) +
      ((totend.tv_usec - totstart.tv_usec) / 1000000.0));
    printf("EBBRT ends\n");
    ebbrt::Cpu::Exit(0);
  });
  
}

int main(int argc, char **argv) {

  ARGC = argc;
  ARGV = argv;
  void* status;

  int numFrontEndCpus;
  
  try {
    po::options_description desc("Options");
    desc.add_options()("help,h", "Print usage messages")
    ("numFrontEndCpus", po::value<int>(&numFrontEndCpus)->default_value(1));
  
    po::variables_map vm;

    try {
      po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
      po::notify(vm);
    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      exit(EXIT_FAILURE);
    }
  } catch (std::exception &e) {
    std::cerr << "Unhandled exception while parsing arguments:  " << e.what()
              << ", application will now exit" << std::endl;
    exit(EXIT_FAILURE);
  }

  pthread_t tid = ebbrt::Cpu::EarlyInit((size_t) numFrontEndCpus);
  pthread_join(tid, &status);
  
  std::cout << "FrontEndCPUS" << numFrontEndCpus << std::endl;
  ebbrt::Cpu::Exit(0);
  return 0;
}

#pragma GCC diagnostic pop

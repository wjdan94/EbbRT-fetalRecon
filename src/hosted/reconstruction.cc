//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "reconstruction.h"
#include "irtkReconstruction.h"

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"


void parseInputParameters(int argc, char **argv) {
  try {
    po::options_description desc("Options");
    desc.add_options()("help,h", "Print usage messages")
      ("output,o", 
        po::value<string>(&PARAMETERS.outputName)->required(),
        "Name for the reconstructed volume. Nifti or Analyze format.")
      ("mask,m", 
        po::value<string>(&PARAMETERS.maskName), 
        "Binary mask to define the region od interest. Nifti or Analyze format.")
      ("input,i", 
        po::value<vector<string>>(&PARAMETERS.inputStacks)->multitoken(),
        "[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format.")
      ("transformation,t",
        po::value<vector<string>>(&PARAMETERS.inputTransformations)->multitoken(),
        "The transformations of the input stack to template in \'dof\' format "
        "used in IRTK. Only rough alignment with correct orienation and some "
        "overlap is needed. Use \'id\' for an identity transformation for at "
        "least one stack. The first stack with \'id\' transformation  will be "
        "resampled as template.")
      ("thickness", 
        po::value<vector<double>>(&PARAMETERS.thickness)->multitoken(),
        "[th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z "
        "direction]")
      ("packages,p", 
        po::value<vector<int>>(&PARAMETERS.packages)->multitoken(),
        "Give number of packages used during acquisition for each stack. The "
        "stacks will be split into packages during registration iteration 1 "
        "and then into odd and even slices within each package during "
        "registration iteration 2. The method will then continue with slice to "
        " volume approach. [Default: slice to volume registration only]")
      ("iterations", 
        po::value<int>(&PARAMETERS.iterations)->default_value(1),
        "Number of registration-reconstruction iterations.")
      ("sigma", 
        po::value<double>(&PARAMETERS.sigma)->default_value(20),
        "Stdev for bias field. [Default: 12mm]")
      ("resolution", 
        po::value<double>(&PARAMETERS.resolution)->default_value(0.75),
        "Isotropic resolution of the volume. [Default: 0.75mm]")
      ("multires", 
        po::value<int>(&PARAMETERS.levels)->default_value(3),
        "Multiresolution smooting with given number of levels. [Default: 3]")
      ("average", 
        po::value<double>(&PARAMETERS.averageValue)->default_value(700),
        "Average intensity value for stacks [Default: 700]")
      ("delta", 
        po::value<double>(&PARAMETERS.delta)->default_value(150),
        "Parameter to define what is an edge. [Default: 150]")
      ("lambda", 
        po::value<double>(&PARAMETERS.lambda)->default_value(0.02),
        "Smoothing parameter. [Default: 0.02]")
      ("lastIterLambda",
        po::value<double>(&PARAMETERS.lastIterLambda)->default_value(0.01),
        "Smoothing parameter for last iteration. [Default: 0.01]")
      ("smoothMask", 
        po::value<double>(&PARAMETERS.smoothMask)->default_value(4),
        "Smooth the mask to reduce artefacts of manual segmentation. [Default: "
        "4mm]")
      ("globalBiasCorrection",
        po::value<bool>(&PARAMETERS.globalBiasCorrection)->default_value(false),
        "Correct the bias in reconstructed image against previous estimation.")
      ("lowIntensityCutoff",
        po::value<double>(&PARAMETERS.lowIntensityCutoff)->default_value(0.01),
        "Lower intensity threshold for inclusion of voxels in global bias "
        "correction.")
      ("forceExclude", 
        po::value<vector<int>>(&PARAMETERS.forceExcluded)->multitoken(),
        "forceExclude [number of slices] [ind1] ... [indN]  "
        "Force exclusion of slices with these indices.")
      ("noIntensityMatching", 
        po::value<bool>(&PARAMETERS.intensityMatching)->default_value(true),
        "Switch off intensity matching.")
      ("logPrefix", 
        po::value<string>(&PARAMETERS.logId), 
        "Prefix for the log file.")
      ("debug", 
        po::value<bool>(&PARAMETERS.debug)->default_value(false),
        "Debug mode - save intermediate results.")
      ("recIterationsFirst",
        po::value<int>(&PARAMETERS.recIterationsFirst)->default_value(4),
        "Set number of superresolution iterations")
      ("recIterationsLast",
        po::value<int>(&PARAMETERS.recIterationsLast)->default_value(13),
        "Set number of superresolution iterations for the last iteration")
      ("numStacksTuner",
        po::value<unsigned int>(&PARAMETERS.numInputStacksTuner)->default_value(0),
        "Set number of input stacks that are really used (for tuner "
        "evaluation, use only first x)")
      ("noLog", 
        po::value<bool>(&PARAMETERS.noLog)->default_value(false),
        "Do not redirect cout and cerr to log files.")
      ("devices,d", 
        po::value<vector<int>>(&PARAMETERS.devicesToUse)->multitoken(),
        "Select the CP > 3.0 GPUs on which the reconstruction should be "
        "executed. Default: all devices > CP 3.0")
      ("tFolder", po::value<string>(&PARAMETERS.tFolder),
        "[folder] Use existing slice-to-volume transformations to initialize "
        "the reconstruction.")
      ("sFolder", 
        po::value<string>(&PARAMETERS.sFolder),
        "[folder] Use existing registered slices and replace loaded ones "
        "(have to be equally many as loaded from stacks).")
      ("referenceVolume", 
        po::value<string>(&PARAMETERS.referenceVolumeName),
        "Name for an optional reference volume. Will be used as inital "
        "reconstruction.")
      ("T1PackageSize", 
        po::value<unsigned int>(&PARAMETERS.T1PackageSize)->default_value(0),
        "is a test if you can register T1 to T2 using NMI and only one "
        "iteration")
      ("numDevicesToUse", 
        po::value<unsigned int>
        (&PARAMETERS.numDevicesToUse)->default_value(UINT_MAX),
        "sets how many GPU devices to use in case of automatic device "
        "selection. Default is as many as available.")
      ("useCPU", po::bool_switch(&PARAMETERS.useCPU)->default_value(false),
        "use CPU for reconstruction and registration; performs superresolution "
        "and robust statistics on CPU. Default is using the GPU")
      ("useCPUReg", 
        po::bool_switch(&PARAMETERS.useCPUReg)->default_value(true),
        "use CPU for more flexible CPU registration; performs superresolution "
        "and robust statistics on GPU. [default, best result]")
      ("useAutoTemplate",
        po::bool_switch(&PARAMETERS.useAutoTemplate)->default_value(false),
        "select 3D registration template stack automatically with matrix rank "
        "method.")
      ("useSINCPSF", 
        po::bool_switch(&PARAMETERS.useSINCPSF)->default_value(false), 
        "use a more MRI like SINC point spread function (PSF) Will " 
        "be in plane sinc (Bartlett) and through plane Gaussian.")
      ("disableBiasCorrection",
        po::bool_switch(&PARAMETERS.disableBiasCorr)->default_value(false),
        "disable bias field correction for cases with little or no bias field "
        "inhomogenities (makes it faster but less reliable for stron intensity "
        "bias)")
      ("numThreads", 
        po::value<int>(&PARAMETERS.numThreads)->default_value(1),
        "Number of CPU threads to run for TBB")
      ("numFrontEndCpus", 
        po::value<int>(&PARAMETERS.numFrontendCPUs)->default_value(1),
        "Number of front-end EbbRT nodes")
      ("numNodes", 
        po::value<int>(&PARAMETERS.numBackendNodes)->default_value(1),
        "Number of back-end EbbRT nodes");

    po::variables_map vm;
    try {
      po::store(po::command_line_parser(argc, argv)
          .options(desc).allow_unregistered().run(), vm);
      if (vm.count("help")) {
        std::cout << "Application to perform reconstruction of volumetric MRI "
                     "from thick slices."
                  << std::endl
                  << desc 
                  << std::endl;
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
}

void AppMain() {
  std::cout << "AppMain()" << std::endl;

  auto reconstruction = irtkReconstruction::Create();
  reconstruction->setParameters(PARAMETERS);
}

int main(int argc, char **argv) {
  void* status;

  EXEC_NAME = argv[0];
  parseInputParameters(argc, argv);

  pthread_t tid = ebbrt::Cpu::EarlyInit((size_t) PARAMETERS.numFrontendCPUs);
  pthread_join(tid, &status);
  
  ebbrt::Cpu::Exit(EXIT_SUCCESS);
  return EXIT_SUCCESS;
}

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
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
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

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

#ifndef APPS_IRTKRECONSTRUCTION_EBB_H_
#define APPS_IRTKRECONSTRUCTION_EBB_H_

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>

#include <vector>

#include <unordered_map>

#include <ebbrt/EbbAllocator.h>
#include <ebbrt/Future.h>
#include <ebbrt/Message.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/StaticIOBuf.h>

#include "utils.h"

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


class irtkReconstructionEbb : public ebbrt::Messagable<irtkReconstructionEbb>, public irtkObject
{ 
private:
  std::mutex m_;
  std::unordered_map<uint32_t, ebbrt::Promise<void>> promise_map_;
  uint32_t id_{0};
  std::vector<ebbrt::Messenger::NetworkId> nids;
  ebbrt::Promise<void> nodesinit;
  ebbrt::Promise<void> mypromise;
  // this is used to save and load context
  ebbrt::EventManager::EventContext* emec{nullptr};
  ebbrt::EventManager::EventContext* emec2{nullptr};
  int numNodes;

public:
  ebbrt::Promise<int> testFuture;
  ebbrt::Promise<int> gaussianreconFuture;
  vector<irtkRealImage> _slices_resampled;
  int _numThreads;
  int _start, _end, _diff, _slices_size, _stack_factor_size;
  int reconRecv, _tnum, _ttnum;
  double tmin, tmax, _tsigma, tmix;
  double _tsum, _tden, _tsum2, _tden2, _tmaxs, _tmins;
  double _ttsum, _ttsum2, _ttden, _ttden2;
  double _msigma, _mmix, _mnum, _mmin, _mmax;
  double _sscalenum, _sscaleden;
  size_t bytesTotal;
  
  /// Transformations
  vector<irtkRigidTransformation> _transformations_gpu;
  /// Indicator whether slice has an overlap with volumetric mask

  vector<bool> _slice_inside_gpu;
  
  //VOLUME
  /// Reconstructed volume
  irtkGenericImage<float> _reconstructed_gpu;

  /// Flag to say whether the template volume has been created
  bool _template_created;


  //pointers for messsage
  int* gaussreconptr = NULL;
  double* gaussreconptr2 = NULL;

  /// Flag to say whether we have a mask
  bool _have_mask;
  /// Weights for Gaussian reconstruction
  irtkRealImage _volume_weights;
  /// Weights for regularization
  irtkRealImage _confidence_map;
  irtkRealImage _addon;
  
  //EM algorithm
  /// Variance for inlier voxel errors
  double _sigma_cpu;
  float _sigma_gpu;
  /// Proportion of inlier voxels
  double _mix_cpu;
  float _mix_gpu;
  /// Uniform distribution for outlier voxels
  double _m_cpu;
  float _m_gpu;
  /// Mean for inlier slice errors
  double _mean_s_cpu;
  float _mean_s_gpu;
  /// Variance for inlier slice errors
  double _sigma_s_cpu;
  float _sigma_s_gpu;
  /// Mean for outlier slice errors
  double _mean_s2_cpu;
  float _mean_s2_gpu;
  /// Variance for outlier slice errors
  double _sigma_s2_cpu;
  float _sigma_s2_gpu;
  /// Proportion of inlier slices
  double _mix_s_cpu;
  float _mix_s_gpu;
  /// Step size for likelihood calculation
  double _step;
  /// Voxel posteriors
  vector<irtkRealImage> _weights;
  ///Slice posteriors
  vector<double> _slice_weight_cpu;
  vector<float> _slice_weight_gpu;
  vector<double> slice_potential;
  
  //Bias field
  ///Variance for bias field
  double _sigma_bias;
  /// Slice-dependent bias fields
  vector<irtkRealImage> _bias;

  ///Slice-dependent scales
  vector<double> _scale_cpu;
  vector<float> _scale_gpu;


  //use sinc like function or not
  bool _use_SINC;

  ///Intensity min and max
  double _max_intensity;
  double _min_intensity;

  //Gradient descent and regulatization parameters
  ///Step for gradient descent
  double _alpha;
  ///Determine what is en edge in edge-preserving smoothing
  double _delta;
  ///Amount of smoothing
  double _lambda;
  ///Average voxel wights to modulate parameter alpha
  double _average_volume_weight;


  //global bias field correction
  ///low intensity cutoff for bias field estimation
  double _low_intensity_cutoff;

  //to restore original signal intensity of the MRI slices
  double _average_value;
  
  //forced excluded slices
  vector<int> _force_excluded;

  //slices identify as too small to be used
  vector<int> _small_slices;

  /// use adaptive or non-adaptive regularisation (default:false)
  bool _adaptive;

  //utility
  ///Debug mode
  bool _debug;


  //Probability density functions
  ///Zero-mean Gaussian PDF
  inline double G(double x, double s);
  ///Uniform PDF
  inline double M(double m);

  int _directions[13][3];

  //Reconstruction* reconstructionGPU;

  bool _useCPUReg;
  bool _useCPU;
  bool _debugGPU;

  static ebbrt::EbbRef<irtkReconstructionEbb>
  Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

  static irtkReconstructionEbb& HandleFault(ebbrt::EbbId id);

  irtkReconstructionEbb(ebbrt::EbbId ebbid);

  ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);

  void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                      std::unique_ptr<ebbrt::IOBuf>&& buffer);
  void SendRecon(int iterations);
  void RunRecon(int iterations, double delta, double lastIterLambda, int rec_iterations_first, int rec_iterations_last, bool intensity_matching, double lambda, int levels);
  
  void Print(ebbrt::Messenger::NetworkId nid, const char* str);
  std::unique_ptr<ebbrt::MutUniqueIOBuf> SerializeSlices();
  std::unique_ptr<ebbrt::MutUniqueIOBuf> SerializeMask();
  std::unique_ptr<ebbrt::MutUniqueIOBuf> SerializeReconstructed();
  std::unique_ptr<ebbrt::MutUniqueIOBuf> SerializeTransformations();
  void DeserializeSlice(ebbrt::IOBuf::DataPointer& dp, irtkRealImage& tmp);
  void DeserializeTransformations(ebbrt::IOBuf::DataPointer& dp, irtkRigidTransformation& tmp);

  //Structures to store the matrix of transformation between volume and slices
  std::vector<SLICECOEFFS> _volcoeffs;
  //std::vector<std::vector<std::vector<std::vector<SLICEINFO> > > >_invertvolcoeffs;
  std::vector<std::vector<std::vector<std::vector<std::vector<SLICEINFO> > > > >_invertvolcoeffs;

  vector<irtkRealImage> _slices;
  //vector<irtkRealImage> _test_slices;
  irtkRealImage _reconstructed;
  irtkRealImage _reconstructed_temp;
  vector<irtkRigidTransformation> _transformations;
  vector<double> _slices_regCertainty;
  //vector<bool> _slice_inside_cpu;
  vector<int> _slice_inside_cpu;

  //SLICES
  /// Slices
  vector<irtkRealImage> _simulated_slices;
  vector<irtkRealImage> _simulated_weights;
  vector<irtkRealImage> _simulated_inside;
  
  vector<float> _stack_factor;
  vector<int> _stack_index;
  vector<int> _voxel_num;
  
  ///global bias correction flag
  bool _global_bias_correction;

  /// Volume mask
  irtkRealImage _mask;

  ///Quality factor - higher means slower and better
  double _quality_factor;

  ///Constructor
  //irtkReconstructionEbb(std::vector<int> dev, bool useCPUReg, bool useCPU = false);
  ///Destructor
  ~irtkReconstructionEbb();

  ///Create zero image as a template for reconstructed volume
  double CreateTemplate(irtkRealImage stack,
    double resolution = 0);

  ///If template image has been masked instead of creating the mask in separate
  ///file, this function can be used to create mask from the template image
  irtkRealImage CreateMask(irtkRealImage image);

  ///Remember volumetric mask and smooth it if necessary
  void SetMask(irtkRealImage * mask, double sigma, double threshold = 0.5);


  void SaveProbabilityMap(int i);

  //std::unique_ptr<ebbrt::MutUniqueIOBuf> serializeImageI2W(irtkRealImage ri, int i, int j);
      
  void setNumNodes(int i);
  void setNumThreads(int i);
  void addNid(ebbrt::Messenger::NetworkId nid);
  ebbrt::Future<void> waitNodes();
  ebbrt::Future<void> waitReceive();

  void CenterStacks(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    int templateNumber);

  //Create average image from the stacks and volumetric transformations
  irtkRealImage CreateAverage(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations);

  ///Crop image according to the mask
  void CropImage(irtkRealImage& image,
    irtkRealImage& mask);

  /// Transform and resample mask to the space of the image
  void TransformMask(irtkRealImage& image,
    irtkRealImage& mask,
    irtkRigidTransformation& transformation);

  /// Rescale image ignoring negative values
  void Rescale(irtkRealImage &img, double max);

  ///Calculate initial registrations
  void StackRegistrations(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    int templateNumber, bool useExternalTarget = false);

  ///Create slices from the stacks and slice-dependent transformations from
  ///stack transformations
  void CreateSlicesAndTransformations(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    vector<double>& thickness,
    const vector<irtkRealImage> &probability_maps = vector<irtkRealImage>());
  void SetSlicesAndTransformations(vector<irtkRealImage>& slices,
    vector<irtkRigidTransformation>& slice_transformations,
    vector<int>& stack_ids,
    vector<double>& thickness);
  void ResetSlices(vector<irtkRealImage>& stacks,
    vector<double>& thickness);

  ///Update slices if stacks have changed
  void UpdateSlices(vector<irtkRealImage>& stacks, vector<double>& thickness);

  void GetSlices(vector<irtkRealImage>& second_stacks);

  ///Invert all stack transformation
  void InvertStackTransformations(vector<irtkRigidTransformation>& stack_transformations);

  ///Match stack intensities
  void MatchStackIntensities(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    double averageValue,
    bool together = false);

  ///Match stack intensities with masking
  void MatchStackIntensitiesWithMasking(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    double averageValue,
    bool together = false);

  ///Mask all slices
  void MaskSlices();

  void InitVoxelStruct();
  
  ///Calculate transformation matrix between slices and voxels
  void CoeffInit(int);

  ///Reconstruction using weighted Gaussian PSF
  void GaussianReconstruction();

  ///Initialise variables and parameters for EM
  void InitializeEM();

  ///Initialise values of variables and parameters for EM
  void InitializeEMValues();

  ///Initalize robust statistics
  void InitializeRobustStatistics();

  ///Perform E-step 
  void EStep();

  ///Calculate slice-dependent scale
  void Scale();

  ///Calculate slice-dependent bias fields
  void Bias();
  void NormaliseBias(int iter);
  
  
  ///Superresolution
  void Superresolution(int iter);

  ///Calculation of voxel-vise robust statistics
  void MStep(int iter);

  ///Edge-preserving regularization
  void Regularization(int iter);

  void AdaptiveRegularization1(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original);
  
  void AdaptiveRegularization2(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original);
  
  ///Edge-preserving regularization with confidence map
  void AdaptiveRegularization(int iter, irtkRealImage& original);

  ///Slice to volume registrations
  void SliceToVolumeRegistration();

  ///Correct bias in the reconstructed volume
  void BiasCorrectVolume(irtkRealImage& original);

  ///Mask the volume
  void MaskVolume();
  void MaskImage(irtkRealImage& image, double padding = -1);


  ///Save slices
  void SaveSlices();
  void SlicesInfo(const char* filename);

  ///Save weights
  void SaveWeights();

  ///Save transformations
  void SaveTransformations();
  void GetTransformations(vector<irtkRigidTransformation> &transformations);
  void SetTransformations(vector<irtkRigidTransformation> &transformations);

  ///Save confidence map
  void SaveConfidenceMap();

  ///Save bias field
  void SaveBiasFields();

  void printvolcoeffs();
  ///Remember stdev for bias field
  inline void SetSigma(double sigma);

  ///Return reconstructed volume
  inline irtkRealImage GetReconstructed();
  void SetReconstructed(irtkRealImage &reconstructed);

  ///Return resampled mask
  inline irtkRealImage GetMask();

  ///Set smoothing parameters
  inline void SetSmoothingParameters(double delta, double lambda);

  inline double SumRecon();

  ///use in-plane sinc like PSF
  inline void useSINCPSF();

  ///Use faster lower quality reconstruction
  inline void SpeedupOn();

  ///Use slower better quality reconstruction
  inline void SpeedupOff();

  ///Switch on global bias correction
  inline void GlobalBiasCorrectionOn();

  ///Switch off global bias correction
  inline void GlobalBiasCorrectionOff();

  ///Set lower threshold for low intensity cutoff during bias estimation
  inline void SetLowIntensityCutoff(double cutoff);

  ///Set slices which need to be excluded by default
  inline void SetForceExcludedSlices(vector<int>& force_excluded);


  //utility
  ///Save intermediate results
  inline void DebugOn();
  ///Do not save intermediate results
  inline void DebugOff();

  inline void UseAdaptiveRegularisation();

  ///Write included/excluded/outside slices
  void Evaluate(int iter);

  /// Read Transformations
  void ReadTransformation(char* folder);

  /// Read and replace Slices
//  void replaceSlices(string folder);

  //To recover original scaling
  ///Restore slice intensities to their original values
  void RestoreSliceIntensities();
  ///Scale volume to match the slice intensities
  void ScaleVolume();

  ///To compare how simulation from the reconstructed volume matches the original stacks
  void SimulateStacks(vector<irtkRealImage>& stacks);

  void SimulateSlices(bool);

  ///Puts origin of the image into origin of world coordinates
  static void ResetOrigin(irtkGreyImage &image,
    irtkRigidTransformation& transformation);

  ///Packages to volume registrations
  void PackageToVolume(vector<irtkRealImage>& stacks,
    vector<int> &pack_num,
    bool evenodd = false,
    bool half = false,
    int half_iter = 1);

  ///Splits stacks into packages
  void SplitImage(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks);
  ///Splits stacks into packages and each package into even and odd slices
  void SplitImageEvenOdd(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks);
  ///Splits image into top and bottom half roi according to z coordinate
  void HalfImage(irtkRealImage image,
    vector<irtkRealImage>& stacks);
  ///Splits stacks into packages and each package into even and odd slices and top and bottom roi
  void SplitImageEvenOddHalf(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks,
    int iter = 1);

  ///sync GPU with data
  //TODO distribute in documented functions above
  irtkRealImage externalRegistrationTargetImage;
  void generatePSFVolume();
  static void ResetOrigin(irtkRealImage &image, irtkRigidTransformation& transformation);

  void PrepareRegistrationSlices();
  //friend class ParallelEStep;
  //friend class ParallelAverage;
  //friend class ParallelSimulateSlices;
  //friend class ParallelStackRegistrations;
  //friend class ParallelScale;
};

inline double irtkReconstructionEbb::G(double x, double s)
{
  return _step*exp(-x*x / (2 * s)) / (sqrt(6.28*s));
}

inline double irtkReconstructionEbb::M(double m)
{
  return m*_step;
}

inline irtkRealImage irtkReconstructionEbb::GetReconstructed()
{
  return _reconstructed;
}

inline irtkRealImage irtkReconstructionEbb::GetMask()
{
  return _mask;
}

inline void irtkReconstructionEbb::DebugOn()
{
  _debug = true;
//  cout << "Debug mode." << endl;
}

inline void irtkReconstructionEbb::UseAdaptiveRegularisation()
{
  _adaptive = true;
}

inline void irtkReconstructionEbb::DebugOff()
{
  _debug = false;
}

inline void irtkReconstructionEbb::SetSigma(double sigma)
{
  _sigma_bias = sigma;
  //cout << "_sigma_bias = " << sigma << endl;
}

inline void irtkReconstructionEbb::useSINCPSF()
{
  _use_SINC = true;
}


inline void irtkReconstructionEbb::SpeedupOn()
{
  _quality_factor = 1;
}

inline void irtkReconstructionEbb::SpeedupOff()
{
  _quality_factor = 2;
}

inline void irtkReconstructionEbb::GlobalBiasCorrectionOn()
{
  _global_bias_correction = true;
//  cout << "_global_bias_correction = true " << endl;
}

inline void irtkReconstructionEbb::GlobalBiasCorrectionOff()
{
  _global_bias_correction = false;
//  cout << "_global_bias_correction = false " << endl;
}

inline void irtkReconstructionEbb::SetLowIntensityCutoff(double cutoff)
{
  if (cutoff > 1) cutoff = 1;
  if (cutoff < 0) cutoff = 0;
  _low_intensity_cutoff = cutoff;
  //cout<<"Setting low intensity cutoff for bias correction to "<<_low_intensity_cutoff<<" of the maximum intensity."<<endl;
}


inline void irtkReconstructionEbb::SetSmoothingParameters(double delta, double lambda)
{
  _delta = delta;
  _lambda = lambda*delta*delta;
  _alpha = 0.05 / lambda;
  if (_alpha > 1) _alpha = 1;
  //cout << "_delta = " << _delta << " _lambda = " << _lambda << " _alpha = " << _alpha << endl;
}

inline void irtkReconstructionEbb::SetForceExcludedSlices(vector<int>& force_excluded)
{
  _force_excluded = force_excluded;
}

inline double irtkReconstructionEbb::SumRecon() {
  float sum = 0.0;
  irtkRealPixel *ap = _reconstructed.GetPointerToVoxels();

  for (int j = 0; j < _reconstructed.GetNumberOfVoxels(); j++) {
      sum += (float)*ap;
    ap++;
  }
  return (double)sum;
}

#endif

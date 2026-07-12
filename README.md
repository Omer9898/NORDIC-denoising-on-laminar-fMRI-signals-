# NORDIC-denoising-on-laminar-fMRI-signals-
Mastetarbeit - omer


From Here:
Post-NORDIC Laminar fMRI Pipeline
This repository documents the full processing and quality-control workflow used to prepare, validate, and analyze post-NORDIC single-subject laminar fMRI data in native EPI space. The project was designed to keep the post-NORDIC analysis comparable to the original pre-NORDIC workflow while minimizing unnecessary interpolation, smoothing, and geometry changes.
Overview
The workflow was built in sequential stages:
	Inspect the post-NORDIC outputs and compare them against the pre-NORDIC reference image.
	Correct NIfTI header inconsistencies without changing voxel values or image geometry.
	Validate that the post-NORDIC data remained on the same native grid as the pre-NORDIC reference.
	Confirm that the first volume of the post-NORDIC 4D series matched the pre-NORDIC reference volume.
	Run first-level GLM separately for run 1, run 2, and run 3 in native EPI space.
	Add ROI-based functional mask quality control in anatomy space.
	Compute whole-brain temporal signal-to-noise ratio (tSNR) before and after NORDIC for quality assessment.
The overall design was intentionally conservative because the data are laminar fMRI. Native-space analysis was preferred, spatial smoothing was avoided, and only final statistical maps were intended for optional downstream warping.
Header correction
A key early step was correcting NIfTI header inconsistencies. Even when the voxel data and affine matrix are correct, mismatches in qform and sform codes can cause software to interpret image geometry inconsistently. This can lead to confusion in later registration, comparison, or analysis steps.
To standardize the files, the post-NORDIC headers were adjusted so that:
	the affine matrix remained unchanged,
	the voxel grid remained unchanged,
	the header codes matched the trusted reference image,
	and the image became consistent for downstream software without resampling the actual data.
This was a metadata correction only. It did not alter the signal values, voxel positions, or time series.
Grid validation
After header correction, the next step was to verify that the post-NORDIC data still matched the original native-space reference grid.
The following properties were checked:
	image shape,
	voxel spacing,
	affine matrix,
	header consistency.
Because the post-NORDIC file was a full 4D time series while the reference image was a single 3D volume, the comparison was performed between the first post-NORDIC volume and the pre-NORDIC reference volume. This showed that the geometry matched exactly.
That result was important because it demonstrated that the NORDIC output remained aligned to the original native EPI space. Therefore, the 4D post-NORDIC series could be used directly for first-level modeling without re-registering the full series.
GLM analysis
The first-level GLM was run separately for run 1, run 2, and run 3 using the full 4D post-NORDIC magnitude series in native EPI space.
The model mirrored the intended SPM-style setup:
	canonical SPM HRF,
	AR(1) noise model,
	cosine drift terms,
	high-pass filtering,
	no spatial smoothing,
	native-space fitting only.
This strategy was chosen to preserve laminar specificity and avoid blurring across cortical depth. A single 3D volume cannot support GLM fitting, so the analysis used the complete 4D time series for each run.
The event timing for each run was inserted directly into the script as a pandas table with three columns:
	onset
	duration
	trial_type
The repetition time (TR) was recovered from the SPM model information:
	T = 16
	dt = 0.1875 s
	TR = 16 × 0.1875 = 3.0 s
For each run, the pipeline produced:
	a design matrix table,
	the event table used in fitting,
	effect-size maps,
	t-statistic maps,
	z-score maps,
	and a basic quality-control tSNR summary.
Functional mask workflow
A separate quality-control step was added to verify that the left and right posterior insula ROI masks were correctly positioned in anatomy space.
Why the functional mask was needed
The aim of this step was not to perform final signal extraction yet, but to visually confirm that the ROI masks were correctly aligned with the anatomy-space functional reference image for each run.
This was especially useful because ROI masks and target images do not always share the same:
	matrix size,
	voxel resolution,
	affine orientation,
	or sampling grid.
Therefore, direct overlay without checking alignment can be misleading.
How the mask was applied
For each run, the anatomy-space mean functional image was used as the grayscale background image. The left and right ROI masks were then loaded and resampled to the reference image grid using nearest-neighbor interpolation.
Nearest-neighbor interpolation was chosen because ROI masks are binary or categorical images. This preserves the mask labels and prevents the creation of artificial partial values at the boundaries.
After resampling:
	the left ROI was converted to a binary mask,
	the right ROI was converted to a binary mask,
	and the two masks were combined into a single overlay image for visualization.
This overlay was then plotted on top of the anatomy-space functional image and saved as a PNG file for each run.
Meaning of the mask outputs
During development, two kinds of outputs were generated:
	Mask files: these contained only the ROI definition itself after resampling to the target image grid.
	Masked image files: these contained the original anatomy-space image after multiplication by the mask, so that only voxels inside the ROI remained visible.
The distinction is simple:
	the mask file tells where the ROI is,
	the masked image shows what the image looks like after the ROI has been applied.
For final visual quality control, the simpler and more useful output was the overlay PNG, because it allowed direct inspection of whether the ROI location was anatomically plausible.
tSNR quality control
Temporal signal-to-noise ratio (tSNR) was added as a before-versus-after quality-control measure to evaluate the effect of NORDIC denoising on the 4D functional time series.
Why tSNR was used
tSNR is a standard fMRI quality metric that reflects temporal stability. For each voxel, it is defined as the temporal mean divided by the temporal standard deviation:
tSNR(v)=μ_t (v)/σ_t (v)
where:
	μ_t (v) is the mean signal across time,
	σ_t (v) is the temporal standard deviation of the same voxel.
Higher tSNR indicates that the voxel signal is more stable over time relative to temporal noise.
Because NORDIC is intended to reduce thermal noise in fMRI data, before-versus-after tSNR comparison is an appropriate first-line quality-control analysis.
How tSNR was computed
The analysis used the full 4D magnitude series before and after NORDIC, and it was performed at the whole-brain level.
For each run:
	the 4D pre-NORDIC magnitude image was loaded,
	the 4D post-NORDIC magnitude image was loaded,
	voxelwise temporal mean and temporal standard deviation were computed,
	valid voxels were defined as finite voxels with positive mean and positive temporal standard deviation,
	voxelwise tSNR was computed,
	a whole-brain mean tSNR value was extracted.
The resulting values were then compared across run 1, run 2, and run 3, and visualized as a before-versus-after paired plot.
What tSNR comparison answers
This comparison answers a focused quality-control question:
	Did temporal stability improve after NORDIC?
It does not directly answer whether statistical activation improved, but it provides important evidence about whether the denoised 4D data became more stable in time.
What was not yet comparable
At one stage, GLM-derived beta-SNR comparison was considered. However, the available files showed that full GLM outputs (beta_*.nii, ResMS.nii, mask.nii) were available for the pre-NORDIC data, while only post-NORDIC magnitude images and related outputs were initially available after NORDIC.
Because a valid beta-SNR comparison requires GLM outputs in both conditions, beta-SNR before-versus-after comparison was deferred until matching post-NORDIC GLM-derived statistics became available.
Therefore, the correct immediate comparison at this stage was:
	before NORDIC vs after NORDIC tSNR on the 4D magnitude time series.
Current outputs by stage
Geometry and preparation
	Header-corrected post-NORDIC 4D magnitude images
	Native-space geometry validation outputs
	Volume comparison checks against the pre-NORDIC reference
GLM outputs (per run)
	design_matrix.csv
	events_runX_used.csv
	maineffectofcognition_effect.nii.gz
	maineffectofcognition_t.nii.gz
	maineffectofcognition_z.nii.gz
	maineffectofpain_effect.nii.gz
	maineffectofpain_t.nii.gz
	maineffectofpain_z.nii.gz
	interactioneffect_effect.nii.gz
	interactioneffect_t.nii.gz
	interactioneffect_z.nii.gz
Functional mask QC
	ROI overlay PNG for each run in anatomy space
	Optional intermediate mask-only NIfTI files
	Optional masked-image NIfTI files
tSNR QC
	whole-brain tSNR summary table
	statistical summary table
	figure comparing before vs after NORDIC across runs
	short English explanation text for reporting
Final interpretation
The complete workflow now includes both structural/geometry validation and functional quality control.
In practical terms, the project established that:
	the post-NORDIC 4D data remained on the correct native grid,
	the NIfTI headers were standardized for reliable downstream handling,
	the first-level GLM could be fit in native EPI space without smoothing,
	the ROI masks could be checked visually in anatomy space,
	and whole-brain tSNR could be used to quantify the impact of NORDIC before and after denoising.
This makes the pipeline suitable for reproducible single-subject laminar fMRI quality control and first-level analysis, while preserving spatial precision and keeping all major processing decisions explicit.

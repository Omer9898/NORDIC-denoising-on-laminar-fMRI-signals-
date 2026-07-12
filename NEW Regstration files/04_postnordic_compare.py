#!/usr/bin/env python3
"""
04_postnordic_compare.py
-------------------------

Extends the validated pre-NORDIC extraction, transformation, and validation
workflow to post-NORDIC functional data.

The script performs three comparisons:

1. Pre-NORDIC reproduced volume vs pre-NORDIC ground-truth
   *_Warped-to-Anat.nii.gz image.

2. Post-NORDIC reproduced volume vs the same pre-NORDIC ground-truth
   *_Warped-to-Anat.nii.gz image.

3. Pre-NORDIC reproduced volume vs post-NORDIC reproduced volume.

The final results are saved in a CSV file.

Key principle:
NORDIC is expected to denoise the functional image without changing its
spatial grid. Therefore, if the pre-NORDIC and post-NORDIC functional images
have the same spatial shape and affine matrix, the already validated
pre-NORDIC transforms can be reused for the post-NORDIC image.

Important:
The script checks that pre-NORDIC and post-NORDIC data share the same spatial
grid before applying the transformation chain. If the grids differ, the script
stops because the pre-NORDIC transforms may no longer be valid.

Interpretation:
- pre_vs_groundtruth:
  Should be very close to perfect if the pre-NORDIC transformation chain
  reproduces the validated result correctly.

- post_vs_groundtruth:
  Is not expected to be exactly 1.0 because NORDIC changes image intensities
  through denoising. A difference here does not automatically indicate a
  registration problem.

- pre_vs_post:
  Quantifies the intensity differences introduced by NORDIC after both images
  have been transformed into anatomical space.

Requirements:
    pip install antspyx nibabel numpy pandas

Usage:
    python 04_postnordic_compare.py \
        --nordic-dir run1/func/NORDIC \
        --prenordic-func run1/func/mag_POCS_r1.nii.gz \
        --ground-truth run1/func/mag_POCS_r1_1000_Warped-to-Anat.nii.gz \
        --reference run1/func/mag_POCS_r1_1000_Warped-to-Anat.nii.gz \
        --raw-index 0 \
        --affine-distcorr mag_POCS_r1_DistCorr_00GenericAffine.mat \
        --warp-distcorr mag_POCS_r1_DistCorr_01Warp.nii.gz \
        --reg2anat custom_reg2anat.txt \
        --motion motion_vol0000.mat \
        --interp lanczosWindowedSinc \
        --out-csv nordic_comparison_summary.csv \
        --out-dir warp_outputs
"""

# Import argparse for command-line argument handling.
import argparse

# Import glob for finding NIfTI files in the NORDIC directory.
import glob

# Import os for path and directory operations.
import os

# Import NumPy for numerical operations and array comparisons.
import numpy as np

# Import NiBabel for reading and writing NIfTI images.
import nibabel as nib

# Import pandas for creating and saving the results table as CSV.
import pandas as pd

# Import ANTsPy for applying registration and distortion-correction transforms.
import ants


# --------------------------------------------------------------------------- #
# Geometry helper functions
# --------------------------------------------------------------------------- #

def grid_signature(img):
    """
    Create a compact spatial-geometry signature for a NIfTI image.

    The signature includes:
    - The first three image dimensions: X, Y, Z
    - Voxel sizes in the first three dimensions
    - The affine transformation matrix

    Parameters
    ----------
    img : nibabel image
        Loaded NIfTI image.

    Returns
    -------
    tuple
        Spatial dimensions, voxel sizes, and rounded affine matrix.
    """

    # Return spatial shape, voxel dimensions, and affine matrix.
    return (
        tuple(img.shape[:3]),
        tuple(round(float(z), 5) for z in img.header.get_zooms()[:3]),
        np.round(img.affine, 4)
    )


def same_grid(img_a, img_b, atol=1e-3):
    """
    Check whether two NIfTI images are defined on the same spatial grid.

    Two images are considered to share the same grid if:
    - Their first three dimensions are identical.
    - Their affine matrices are equal within a specified tolerance.

    Parameters
    ----------
    img_a : nibabel image
        First image.
    img_b : nibabel image
        Second image.
    atol : float
        Absolute tolerance used when comparing affine values.

    Returns
    -------
    bool
        True if both images have matching spatial dimensions and affine matrices.
    """

    # Extract spatial dimensions only; ignore the time dimension if present.
    sa = img_a.shape[:3]
    sb = img_b.shape[:3]

    # Compare image dimensions and affine matrices.
    return sa == sb and np.allclose(img_a.affine, img_b.affine, atol=atol)


# --------------------------------------------------------------------------- #
# Detect the post-NORDIC functional 4D image
# --------------------------------------------------------------------------- #

def detect_functional_4d(nordic_dir, reference_grid_img):
    """
    Detect the post-NORDIC functional 4D NIfTI time series inside a directory.

    A valid candidate must:
    - Be a readable NIfTI file.
    - Be a 4D image with more than one volume.
    - Have the same spatial grid as the pre-NORDIC reference image.
    - Not appear to be a derived map such as noise, residual, mask, mean, or SNR.

    Parameters
    ----------
    nordic_dir : str
        Directory containing NORDIC output NIfTI files.
    reference_grid_img : nibabel image
        Pre-NORDIC functional image used as the reference spatial grid.

    Returns
    -------
    tuple
        Selected post-NORDIC functional path and a description of why it was selected.
    """

    # File-name tokens that suggest a derived image rather than the denoised
    # functional time series.
    reject_tokens = (
        "gfactor", "noise", "resid", "residual", "mask",
        "mean", "std", "snr", "tsnr", "sos", "phase"
    )

    # Store valid candidate files here.
    candidates = []

    # Search for NIfTI files with extensions such as .nii and .nii.gz.
    for path in sorted(glob.glob(os.path.join(nordic_dir, "*.nii*"))):

        # Convert filename to lowercase for robust token matching.
        base = os.path.basename(path).lower()

        # Try loading the candidate image.
        try:
            img = nib.load(path)

        # Skip unreadable or invalid NIfTI files.
        except Exception as e:
            print(f"[detect] Unreadable file skipped: {base} ({e})")
            continue

        # Obtain the number of dimensions.
        ndim = img.ndim

        # Obtain the number of time volumes for a 4D image.
        # A 3D image is treated as a single-volume image.
        nvol = img.shape[3] if ndim == 4 else 1

        # Check whether the image is a real functional time series.
        is_time_series = (ndim == 4 and nvol > 1)

        # Check whether the candidate shares the pre-NORDIC spatial grid.
        grid_ok = same_grid(img, reference_grid_img)

        # Check whether the filename does not look like a derived map.
        name_ok = not any(token in base for token in reject_tokens)

        # Print diagnostic information for every candidate file.
        print(
            f"[detect] {base:<45} "
            f"ndim={ndim} nvol={nvol} "
            f"grid_match={grid_ok} name_ok={name_ok}"
        )

        # Keep files that satisfy all selection rules.
        if is_time_series and grid_ok and name_ok:
            candidates.append((path, nvol))

    # Stop if no valid post-NORDIC functional image was found.
    if not candidates:
        raise FileNotFoundError(
            "No post-NORDIC functional 4D image was found. "
            "Expected a non-derived 4D time series on the same spatial grid "
            "as the pre-NORDIC functional image."
        )

    # Warn the user if more than one file satisfies the criteria.
    if len(candidates) > 1:
        candidate_names = [os.path.basename(path) for path, _ in candidates]
        print(f"[detect] WARNING: Multiple candidates found: {candidate_names}")

    # Select the first candidate after alphabetical sorting.
    path, nvol = candidates[0]

    # Return the selected file path and a short selection explanation.
    return (
        path,
        f"4D time series ({nvol} volumes) on matching grid; non-derived filename"
    )


# --------------------------------------------------------------------------- #
# Extract one 3D volume while preserving geometry
# --------------------------------------------------------------------------- #

def extract_volume(func_path, index, out_path):
    """
    Extract one 3D volume from a 4D NIfTI functional time series.

    The extracted volume preserves:
    - Original affine matrix
    - qform and sform metadata
    - Voxel sizes
    - Original data type

    Parameters
    ----------
    func_path : str
        Path to the input 4D functional NIfTI file.
    index : int
        Zero-based time-volume index to extract.
    out_path : str
        Path where the extracted 3D NIfTI image will be saved.

    Returns
    -------
    str
        Path to the saved 3D volume.
    """

    # Load the input functional image.
    img = nib.load(func_path)

    # Confirm that the input image is 4D.
    if img.ndim != 4:
        raise ValueError(f"{func_path} is not a 4D image; shape is {img.shape}")

    # Validate that the requested index exists in the time dimension.
    if not (0 <= index < img.shape[3]):
        raise IndexError(
            f"Index {index} is out of range. "
            f"Valid indices: 0 to {img.shape[3] - 1}"
        )

    # Extract the requested 3D image volume.
    vol = np.asarray(img.dataobj[..., index])

    # Copy the original image header.
    hdr = img.header.copy()

    # Update the header from 4D shape to the new 3D shape.
    hdr.set_data_shape(vol.shape)

    # Preserve qform spatial metadata and its code.
    hdr.set_qform(img.get_qform(), code=int(img.header["qform_code"]))

    # Preserve sform spatial metadata and its code.
    hdr.set_sform(img.get_sform(), code=int(img.header["sform_code"]))

    # Preserve X, Y, and Z voxel sizes.
    hdr.set_zooms(list(img.header.get_zooms()[:3]))

    # Create the output 3D NIfTI image using the original affine matrix.
    out = nib.Nifti1Image(vol, img.affine, header=hdr)

    # Preserve the original data type, for example float32 or int16.
    out.set_data_dtype(img.get_data_dtype())

    # Save the extracted volume.
    nib.save(out, out_path)

    # Return the output path.
    return out_path


# --------------------------------------------------------------------------- #
# Build and apply the validated transformation chain
# --------------------------------------------------------------------------- #

def build_chain(reg2anat, warp_distcorr, affine_distcorr, motion=None):
    """
    Build the ordered ANTs transformation list.

    The transformation chain consists of:
    1. Functional-to-anatomical registration transform.
    2. Nonlinear distortion-correction warp.
    3. Affine distortion-correction transform.
    4. Optional volume-specific motion-correction transform.

    Parameters
    ----------
    reg2anat : str
        Transform mapping functional image space toward anatomical space.
    warp_distcorr : str
        Nonlinear distortion-correction deformation field.
    affine_distcorr : str
        Affine distortion-correction transform.
    motion : str or None
        Optional motion-correction transform for the selected volume.

    Returns
    -------
    list
        Ordered list of transform paths for ANTs.
    """

    # Create the validated transform chain.
    # ANTs applies transforms according to its transform-list conventions.
    chain = [reg2anat, warp_distcorr, affine_distcorr]

    # Add optional motion correction as the transform closest to the moving image.
    if motion:
        chain = chain + [motion]

    # Return the ordered transform list.
    return chain


def apply_chain(moving_path, reference_path, chain, out_path, interp="linear"):
    """
    Apply a transformation chain to a moving image using ANTsPy.

    Parameters
    ----------
    moving_path : str
        Path to the input image that will be transformed.
    reference_path : str
        Path to the anatomical-space reference image defining the output grid.
    chain : list
        Ordered list of transformation files.
    out_path : str
        Path where the transformed result will be saved.
    interp : str
        Interpolation method used during resampling.
        Examples: "linear", "nearestNeighbor", or "lanczosWindowedSinc".

    Returns
    -------
    str
        Path to the saved transformed image.
    """

    # Load the image that will be transformed.
    mov = ants.image_read(moving_path)

    # Load the fixed anatomical-space reference image.
    ref = ants.image_read(reference_path)

    # Apply the transformation list and resample the moving image
    # onto the reference image grid.
    out = ants.apply_transforms(
        fixed=ref,
        moving=mov,
        transformlist=chain,
        interpolator=interp
    )

    # Save the transformed image.
    ants.image_write(out, out_path)

    # Return the saved output path.
    return out_path


# --------------------------------------------------------------------------- #
# Similarity metrics
# --------------------------------------------------------------------------- #

def compare(reproduced_path, target_path):
    """
    Compare two NIfTI images using Pearson correlation, RMSE, and NRMSE.

    The comparison is performed over voxels where at least one image
    has non-zero intensity. Shared zero-valued background voxels are excluded.

    Parameters
    ----------
    reproduced_path : str
        Path to the first image.
    target_path : str
        Path to the second comparison image.

    Returns
    -------
    tuple
        Pearson correlation coefficient, RMSE, and NRMSE.
    """

    # Load both NIfTI images.
    a = nib.load(reproduced_path)
    b = nib.load(target_path)

    # Verify that both images have the same dimensions.
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Convert image values to flattened float64 arrays.
    x = a.get_fdata(dtype=np.float64).ravel()
    y = b.get_fdata(dtype=np.float64).ravel()

    # Keep only voxels with finite values in both images.
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    # Exclude shared zero-valued background voxels.
    support_mask = (x != 0) | (y != 0)
    xs = x[support_mask]
    ys = y[support_mask]

    # Ensure that at least one meaningful voxel is available.
    if xs.size == 0:
        raise ValueError("No valid non-zero voxels are available for comparison.")

    # Calculate Pearson correlation coefficient.
    r = float(np.corrcoef(xs, ys)[0, 1])

    # Calculate root mean squared error.
    rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))

    # Normalize RMSE by the target-image intensity range.
    nrmse = rmse / (ys.max() - ys.min() + 1e-12)

    # Return the similarity metrics.
    return r, rmse, nrmse


def geom_fields(path):
    """
    Extract geometry-related metadata fields from a NIfTI image.

    Parameters
    ----------
    path : str
        Path to a NIfTI image.

    Returns
    -------
    dict
        Image shape, voxel sizes, qform code, and sform code.
    """

    # Load the image.
    img = nib.load(path)

    # Return metadata in a CSV-friendly format.
    return {
        "shape": "x".join(str(s) for s in img.shape),
        "zooms": ",".join(f"{z:.4f}" for z in img.header.get_zooms()[:3]),
        "qform_code": int(img.header["qform_code"]),
        "sform_code": int(img.header["sform_code"]),
    }


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #

def main():
    """
    Run the complete pre-NORDIC and post-NORDIC comparison workflow.

    Workflow:
    1. Parse command-line arguments.
    2. Detect the post-NORDIC functional 4D image.
    3. Confirm spatial-grid identity with the pre-NORDIC image.
    4. Extract the same 3D volume from both datasets.
    5. Apply the validated transform chain to both volumes.
    6. Compute three image comparisons.
    7. Save results as a CSV file.
    """

    # Create the command-line argument parser.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Directory containing NORDIC output files.
    ap.add_argument(
        "--nordic-dir",
        required=True,
        help="Directory containing post-NORDIC NIfTI output files."
    )

    # Path to the original pre-NORDIC functional 4D time series.
    ap.add_argument(
        "--prenordic-func",
        required=True,
        help="Path to the original pre-NORDIC functional 4D NIfTI image."
    )

    # Validated pre-NORDIC warped-to-anatomy output used as ground truth.
    ap.add_argument(
        "--ground-truth",
        required=True,
        help="Path to the validated pre-NORDIC *_Warped-to-Anat.nii.gz image."
    )

    # Anatomical reference image defining the target output grid.
    ap.add_argument(
        "--reference",
        required=True,
        help="Anatomical-space reference image defining the output grid."
    )

    # Volume index to extract from both pre- and post-NORDIC time series.
    ap.add_argument(
        "--raw-index",
        type=int,
        required=True,
        help="Zero-based functional-volume index to extract and transform."
    )

    # Distortion-correction affine transform.
    ap.add_argument(
        "--affine-distcorr",
        required=True,
        help="Path to the affine distortion-correction transform."
    )

    # Distortion-correction nonlinear deformation field.
    ap.add_argument(
        "--warp-distcorr",
        required=True,
        help="Path to the nonlinear distortion-correction warp field."
    )

    # Functional-to-anatomical registration transform.
    ap.add_argument(
        "--reg2anat",
        required=True,
        help="Path to the functional-to-anatomical registration transform."
    )

    # Optional volume-specific motion-correction transform.
    ap.add_argument(
        "--motion",
        default=None,
        help="Optional motion-correction transform for the selected volume."
    )

    # Interpolation method for image resampling.
    ap.add_argument(
        "--interp",
        default="linear",
        help="Interpolation method used by ANTs, for example linear or lanczosWindowedSinc."
    )

    # Output CSV file containing the comparison results.
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Path for the output CSV summary file."
    )

    # Directory for extracted and transformed images.
    ap.add_argument(
        "--out-dir",
        default="warp_outputs",
        help="Directory for extracted and warped output volumes."
    )

    # Parse all command-line arguments.
    a = ap.parse_args()

    # Create the output directory if necessary.
    os.makedirs(a.out_dir, exist_ok=True)

    # Load the pre-NORDIC functional image.
    pre_img = nib.load(a.prenordic_func)

    # ----------------------------------------------------------------------- #
    # Step 1: Detect the post-NORDIC functional 4D image.
    # ----------------------------------------------------------------------- #
    post_func, reason = detect_functional_4d(a.nordic_dir, pre_img)

    print(
        f"\n[detect] SELECTED post-NORDIC 4D image: {post_func}"
        f"\n         Reason: {reason}"
    )

    # ----------------------------------------------------------------------- #
    # Step 2: Confirm identical pre-NORDIC and post-NORDIC spatial grids.
    # ----------------------------------------------------------------------- #
    post_img = nib.load(post_func)

    # Stop immediately if post-NORDIC data are not on the same spatial grid.
    if not same_grid(pre_img, post_img):
        raise SystemExit(
            "[STOP] The post-NORDIC grid differs from the pre-NORDIC grid.\n"
            f"  Pre-NORDIC: shape={pre_img.shape[:3]}\n"
            f"  Pre affine:\n{np.round(pre_img.affine, 4)}\n"
            f"  Post-NORDIC: shape={post_img.shape[:3]}\n"
            f"  Post affine:\n{np.round(post_img.affine, 4)}\n"
            "The pre-NORDIC transformation files should not be reused. "
            "Stop and verify the NORDIC output geometry."
        )

    # Warn if the number of volumes is different.
    if pre_img.shape[3] != post_img.shape[3]:
        print(
            f"[warn] Number of volumes differs: "
            f"pre={pre_img.shape[3]}, post={post_img.shape[3]}. "
            "Check that the selected index still corresponds to the intended "
            "functional acquisition volume."
        )

    # Confirm that transform reuse is spatially valid.
    print("[assert] Pre-NORDIC and post-NORDIC grid identity: OK")

    # Store the selected raw-volume index.
    idx = a.raw_index

    # Build the validated transformation chain.
    chain = build_chain(
        a.reg2anat,
        a.warp_distcorr,
        a.affine_distcorr,
        a.motion
    )

    # Print the ordered transform list.
    print("\n[chain] Transform list:")
    for transform in chain:
        print("  -", transform)

    # ----------------------------------------------------------------------- #
    # Step 3: Extract the same 3D volume from pre- and post-NORDIC datasets.
    # ----------------------------------------------------------------------- #
    pre_vol = extract_volume(
        a.prenordic_func,
        idx,
        os.path.join(a.out_dir, f"pre_vol{idx:04d}.nii.gz")
    )

    post_vol = extract_volume(
        post_func,
        idx,
        os.path.join(a.out_dir, f"post_vol{idx:04d}.nii.gz")
    )

    # ----------------------------------------------------------------------- #
    # Step 4: Apply the same transform chain to both extracted volumes.
    # ----------------------------------------------------------------------- #
    pre_warp = apply_chain(
        pre_vol,
        a.reference,
        chain,
        os.path.join(a.out_dir, f"pre_vol{idx:04d}_Warped-to-Anat.nii.gz"),
        a.interp
    )

    post_warp = apply_chain(
        post_vol,
        a.reference,
        chain,
        os.path.join(a.out_dir, f"post_vol{idx:04d}_Warped-to-Anat.nii.gz"),
        a.interp
    )

    # Obtain common anatomical-space geometry fields for the output CSV.
    gfields = geom_fields(pre_warp)

    # Create a list to store each comparison result.
    rows = []

    def add_comparison(comparison_name, moving_path, target_path):
        """
        Calculate metrics for one image pair and add them to the CSV table.

        Parameters
        ----------
        comparison_name : str
            Descriptive label for the comparison.
        moving_path : str
            Path to the first image.
        target_path : str
            Path to the second image.
        """

        # Calculate similarity metrics.
        r, rmse, nrmse = compare(moving_path, target_path)

        # Add one row to the summary table.
        rows.append({
            "comparison": comparison_name,
            "moving_filename": os.path.basename(moving_path),
            "target_filename": os.path.basename(target_path),
            "raw_index": idx,
            "pearson_r": round(r, 6),
            "rmse": round(rmse, 6),
            "nrmse": round(nrmse, 6),
            "shape": gfields["shape"],
            "zooms": gfields["zooms"],
            "qform_code": gfields["qform_code"],
            "sform_code": gfields["sform_code"],
        })

    # ----------------------------------------------------------------------- #
    # Step 5: Perform all three requested comparisons.
    # ----------------------------------------------------------------------- #

    # Validate pre-NORDIC reproduction against its known ground-truth image.
    add_comparison(
        "pre_vs_groundtruth",
        pre_warp,
        a.ground_truth
    )

    # Measure post-NORDIC intensity differences relative to pre-NORDIC ground truth.
    add_comparison(
        "post_vs_groundtruth",
        post_warp,
        a.ground_truth
    )

    # Quantify the denoising effect in anatomical space.
    add_comparison(
        "pre_vs_post",
        post_warp,
        pre_warp
    )

    # Convert all metric rows to a pandas DataFrame.
    df = pd.DataFrame(rows)

    # Write the results table to CSV without the pandas row index.
    df.to_csv(a.out_csv, index=False)

    # Print the complete results table in the terminal.
    print("\n" + df.to_string(index=False))

    # Confirm CSV output location.
    print(f"\n[csv] Written to: {a.out_csv}")

    # Print concise interpretation guidance.
    print(
        "[note] pre_vs_groundtruth close to 1.0 confirms that the validated "
        "pre-NORDIC reproduction was successful.\n"
        "[note] post_vs_groundtruth below 1.0 usually reflects NORDIC-related "
        "intensity changes, not necessarily a registration error.\n"
        "[note] pre_vs_post quantifies the effect of NORDIC denoising in "
        "anatomical space."
    )


# Run the main workflow only when this file is executed directly.
if __name__ == "__main__":
    main()
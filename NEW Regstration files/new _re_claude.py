#!/usr/bin/env python3
"""
05_registration_diagnosis.py
-----------------------------

Test multiple source-volume indices and interpolation methods to determine
which combination best reproduces a known ground-truth warped-to-anatomical
NIfTI image.

For each selected raw index and interpolation method, the script:

1. Extracts one 3D volume from the original 4D functional NIfTI image.
2. Applies the predefined distortion-correction and registration transforms.
3. Compares the reproduced output with the ground-truth anatomical-space image.
4. Calculates correlation and error metrics.
5. Saves all results in a CSV file.
6. Ranks the tested combinations by correlation on non-zero support voxels.

The best result is the row with the highest r_support value.
"""

# Import argparse for reading command-line arguments.
import argparse

# Import os for file paths and directory creation.
import os

# Import tempfile for creating temporary files that are deleted automatically.
import tempfile

# Import NumPy for numerical operations and similarity metrics.
import numpy as np

# Import NiBabel for reading and writing NIfTI neuroimaging files.
import nibabel as nib

# Import pandas for creating, sorting, and exporting the results table.
import pandas as pd

# Import ANTsPy for applying affine, warp, registration, and motion transforms.
import ants


# --------------------------------------------------------------------------- #
# Default paths and settings
# --------------------------------------------------------------------------- #

# Base directory where the final CSV output will be stored.
BASE_DIR = r"D:/NORDIC_Subject1_project/NEW Regstration files"

# Path to the original 4D functional NIfTI image.
FUNC_PATH = r"D:\Downloads\run1\func\mag_POCS_r1.nii.gz"

# Path to the known ground-truth image in anatomical space.
GROUND_TRUTH = r"D:\Downloads\run1\func\mag_POCS_r1_1000_Warped-to-Anat.nii.gz"

# Reference image that defines the target anatomical-space grid.
REFERENCE = r"D:\Downloads\run1\func\mag_POCS_r1_1000_Warped-to-Anat.nii.gz"

# Affine transformation generated during distortion correction.
AFFINE_DISTCORR = r"D:\Downloads\run1\func\mag_POCS_r1_DistCorr_00GenericAffine.mat"

# Nonlinear warp field generated during distortion correction.
WARP_DISTCORR = r"D:\Downloads\run1\func\mag_POCS_r1_DistCorr_01Warp.nii.gz"

# Functional-to-anatomical registration transform.
REG2ANAT = r"D:\Downloads\run1\func\custom_reg2anat.txt"

# Default CSV path used to save the registration diagnosis results.
OUT_CSV = os.path.join(BASE_DIR, "reg_diagnosis.csv")

# Default source-volume indices to test.
INDICES_DEFAULT = "0,1,2,5,10"

# Default interpolation methods to test.
INTERPS_DEFAULT = "linear,lanczosWindowedSinc"


# --------------------------------------------------------------------------- #
# Extract one 3D volume from a 4D functional dataset
# --------------------------------------------------------------------------- #

def extract(func_img, index, out_path):
    """
    Extract one 3D volume from a loaded 4D functional NIfTI image.

    The output preserves the original image geometry and metadata:
    - Affine matrix
    - qform and sform information
    - Voxel dimensions
    - Original data type

    Parameters
    ----------
    func_img : nibabel image
        Loaded 4D functional NIfTI image.
    index : int
        Zero-based index of the volume to extract.
    out_path : str
        Output path for the extracted 3D NIfTI image.

    Returns
    -------
    str
        Path to the saved 3D volume.
    """

    # Extract the selected 3D volume from the fourth/time dimension.
    vol = np.asarray(func_img.dataobj[..., index])

    # Copy the original header to preserve image metadata.
    hdr = func_img.header.copy()

    # Update the image shape in the header from 4D to 3D.
    hdr.set_data_shape(vol.shape)

    # Preserve the qform spatial transformation matrix and its code.
    hdr.set_qform(
        func_img.get_qform(),
        code=int(func_img.header["qform_code"])
    )

    # Preserve the sform spatial transformation matrix and its code.
    hdr.set_sform(
        func_img.get_sform(),
        code=int(func_img.header["sform_code"])
    )

    # Preserve voxel sizes in X, Y, and Z directions.
    hdr.set_zooms(list(func_img.header.get_zooms()[:3]))

    # Create a new 3D NIfTI image using the original affine matrix.
    out = nib.Nifti1Image(vol, func_img.affine, header=hdr)

    # Preserve the original data type, for example float32 or int16.
    out.set_data_dtype(func_img.get_data_dtype())

    # Save the extracted volume to the specified location.
    nib.save(out, out_path)

    # Return the output path.
    return out_path


# --------------------------------------------------------------------------- #
# Build the ANTs transformation list
# --------------------------------------------------------------------------- #

def build_chain(reg2anat, warp_distcorr, affine_distcorr, motion=None):
    """
    Build the ordered list of spatial transformations for ANTs.

    Parameters
    ----------
    reg2anat : str
        Functional-to-anatomical registration transform.
    warp_distcorr : str
        Nonlinear distortion-correction warp field.
    affine_distcorr : str
        Affine distortion-correction transform.
    motion : str or None
        Optional motion-correction transform for the selected volume.

    Returns
    -------
    list
        Ordered transformation list passed to ants.apply_transforms().
    """

    # Create the standard validated transformation chain.
    chain = [reg2anat, warp_distcorr, affine_distcorr]

    # Add a volume-specific motion transform if it was supplied.
    if motion:
        chain = chain + [motion]

    # Return the full transformation chain.
    return chain


# --------------------------------------------------------------------------- #
# Apply the transformation chain
# --------------------------------------------------------------------------- #

def apply_chain(moving, reference, chain, out_path, interp):
    """
    Apply spatial transformations to a moving NIfTI image using ANTsPy.

    Parameters
    ----------
    moving : str
        Path to the 3D source image to be transformed.
    reference : str
        Path to the anatomical-space reference image defining the target grid.
    chain : list
        Ordered list of transform files.
    out_path : str
        Path where the transformed output image will be saved.
    interp : str
        Interpolation method used during resampling.
        Examples include "linear" and "lanczosWindowedSinc".

    Returns
    -------
    str
        Path to the saved transformed NIfTI image.
    """

    # Apply transformations and resample the moving image into reference space.
    out = ants.apply_transforms(
        fixed=ants.image_read(reference),
        moving=ants.image_read(moving),
        transformlist=chain,
        interpolator=interp
    )

    # Save the transformed output image.
    ants.image_write(out, out_path)

    # Return the output path.
    return out_path


# --------------------------------------------------------------------------- #
# Calculate image-similarity metrics
# --------------------------------------------------------------------------- #

def metrics(reproduced_path, truth_path, mask_path=None):
    """
    Calculate similarity metrics between a reproduced image and ground truth.

    Metrics:
    - r_support: Pearson correlation over the union of non-zero voxels
    - r_full: Pearson correlation over all finite voxels, including background
    - r_mask: Pearson correlation inside an optional user-provided mask
    - RMSE: Root Mean Squared Error over non-zero support
    - NRMSE: Normalized RMSE over non-zero support

    Parameters
    ----------
    reproduced_path : str
        Path to the reproduced transformed image.
    truth_path : str
        Path to the ground-truth image.
    mask_path : str or None
        Optional path to a binary mask in the same target grid.

    Returns
    -------
    tuple
        r_support, r_full, r_mask, RMSE, and NRMSE.
    """

    # Load and flatten the reproduced image into a float64 vector.
    a = nib.load(reproduced_path).get_fdata(dtype=np.float64).ravel()

    # Load and flatten the ground-truth image into a float64 vector.
    b = nib.load(truth_path).get_fdata(dtype=np.float64).ravel()

    # Keep only voxels where both images have finite intensity values.
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]

    # Calculate correlation using all valid voxels, including background.
    r_full = float(np.corrcoef(a, b)[0, 1])

    # Define non-zero support as voxels where either image is not zero.
    # This avoids shared zero background artificially increasing similarity.
    supp = (a != 0) | (b != 0)

    # Calculate correlation over the meaningful non-zero support region.
    r_supp = float(np.corrcoef(a[supp], b[supp])[0, 1])

    # Calculate root mean squared error over the same support region.
    rmse = float(np.sqrt(np.mean((a[supp] - b[supp]) ** 2)))

    # Normalize RMSE by the ground-truth intensity range.
    # The small constant prevents division by zero.
    nrmse = rmse / (b[supp].max() - b[supp].min() + 1e-12)

    # Set masked correlation to NaN when no mask is supplied.
    r_mask = np.nan

    # Compute correlation within an optional mask, if provided.
    if mask_path:

        # Load the mask, flatten it, and align it with finite voxel filtering.
        m = nib.load(mask_path).get_fdata().ravel().astype(bool)[finite]

        # Calculate masked correlation only if the mask includes voxels.
        if m.any():
            r_mask = float(np.corrcoef(a[m], b[m])[0, 1])

    # Return all metrics.
    return r_supp, r_full, r_mask, rmse, nrmse


# --------------------------------------------------------------------------- #
# Main diagnostic workflow
# --------------------------------------------------------------------------- #

def main():
    """
    Test multiple functional-volume indices and interpolation methods.

    The script ranks all tested combinations according to r_support,
    which is Pearson correlation over the union of non-zero support voxels.
    """

    # Create a command-line argument parser.
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose registration settings by testing source-volume indices "
            "and interpolation methods against a ground-truth anatomical-space image."
        )
    )

    # Input 4D functional NIfTI dataset.
    ap.add_argument(
        "--func",
        default=FUNC_PATH,
        help="Path to the source 4D functional NIfTI image."
    )

    # Known ground-truth image in anatomical space.
    ap.add_argument(
        "--ground-truth",
        default=GROUND_TRUTH,
        help="Path to the ground-truth *_Warped-to-Anat.nii.gz image."
    )

    # Reference image defining output anatomical-space geometry.
    ap.add_argument(
        "--reference",
        default=REFERENCE,
        help="Path to the target anatomical-space reference image."
    )

    # Affine transform used for distortion correction.
    ap.add_argument(
        "--affine-distcorr",
        default=AFFINE_DISTCORR,
        help="Path to the affine distortion-correction transform."
    )

    # Nonlinear deformation field used for distortion correction.
    ap.add_argument(
        "--warp-distcorr",
        default=WARP_DISTCORR,
        help="Path to the nonlinear distortion-correction warp field."
    )

    # Functional-to-anatomical registration transform.
    ap.add_argument(
        "--reg2anat",
        default=REG2ANAT,
        help="Path to the functional-to-anatomical registration transform."
    )

    # Optional per-volume motion-correction transform.
    ap.add_argument(
        "--motion",
        default=None,
        help="Optional motion-correction transform."
    )

    # Comma-separated list of source volume indices to test.
    ap.add_argument(
        "--indices",
        default=INDICES_DEFAULT,
        help="Comma-separated 4D source-volume indices, for example: 0,1,2,5,10."
    )

    # Comma-separated list of interpolation methods to test.
    ap.add_argument(
        "--interps",
        default=INTERPS_DEFAULT,
        help="Comma-separated interpolation methods, for example: linear,lanczosWindowedSinc."
    )

    # Optional anatomical-space binary mask for masked Pearson correlation.
    ap.add_argument(
        "--mask",
        default=None,
        help="Optional binary mask image in the same grid as the ground-truth image."
    )

    # Output CSV path for the diagnostic results.
    ap.add_argument(
        "--out-csv",
        default=OUT_CSV,
        help="Output CSV file containing all tested combinations and metrics."
    )

    # Parse command-line arguments.
    args = ap.parse_args()

    # Create the base output directory if it does not already exist.
    os.makedirs(BASE_DIR, exist_ok=True)

    # Load the functional image.
    func = nib.load(args.func)

    # Obtain the number of time volumes if the image is 4D.
    n_t = func.shape[3] if func.ndim == 4 else 0

    # Convert comma-separated indices into a list of integers.
    idxs = [int(s) for s in args.indices.split(",")]

    # Convert comma-separated interpolation names into a clean list of strings.
    interps = [s.strip() for s in args.interps.split(",")]

    # Build the spatial transformation chain.
    chain = build_chain(
        args.reg2anat,
        args.warp_distcorr,
        args.affine_distcorr,
        args.motion
    )

    # Display the selected test settings.
    print(
        f"[info] Functional 4D shape: {func.shape}; "
        f"testing indices: {idxs}; interpolators: {interps}"
    )

    # Report whether a motion transform is included.
    print(
        f"[info] Motion transform: "
        f"{'INCLUDED' if args.motion else 'NONE'}"
    )

    # Print the ordered transformation chain.
    print(f"[chain] {chain}\n")

    # Store one result dictionary for each tested combination.
    rows = []

    # Create a temporary directory for intermediate extracted and warped images.
    # Files inside this directory are automatically removed when processing ends.
    with tempfile.TemporaryDirectory() as td:

        # Test each requested source-volume index.
        for idx in idxs:

            # Skip unavailable indices.
            if not (0 <= idx < n_t):
                print(f"[skip] Index {idx} is out of range 0 to {n_t - 1}")
                continue

            # Extract the current 3D source volume.
            mv = extract(
                func,
                idx,
                os.path.join(td, f"v{idx}.nii.gz")
            )

            # Apply the transform chain using every requested interpolation method.
            for interp in interps:

                # Define a temporary output path for this result.
                rep = os.path.join(td, f"v{idx}_{interp}.nii.gz")

                # Transform the extracted volume into anatomical space.
                apply_chain(
                    mv,
                    args.reference,
                    chain,
                    rep,
                    interp
                )

                # Compare the transformed image against the ground truth.
                r_supp, r_full, r_mask, rmse, nrmse = metrics(
                    rep,
                    args.ground_truth,
                    args.mask
                )

                # Store results for this index/interpolation combination.
                rows.append({
                    "raw_index": idx,
                    "interp": interp,
                    "r_support": round(r_supp, 5),
                    "r_full": round(r_full, 5),
                    "r_mask": round(r_mask, 5) if not np.isnan(r_mask) else "",
                    "rmse": round(rmse, 4),
                    "nrmse": round(nrmse, 5),
                })

                # Print current result to the terminal.
                print(
                    f"Index {idx:>5} | "
                    f"{interp:<20} | "
                    f"r_support={r_supp:.5f} | "
                    f"r_full={r_full:.5f} | "
                    f"nrmse={nrmse:.5f}"
                )

    # Convert all results into a pandas DataFrame and rank by r_support.
    df = pd.DataFrame(rows).sort_values(
        "r_support",
        ascending=False
    )

    # Save the ranked diagnostic results as a CSV file.
    df.to_csv(args.out_csv, index=False)

    # Print the full ranking.
    print("\n[ranked by r_support]")
    print(df.to_string(index=False))

    # Print the location of the saved CSV file.
    print(f"\n[csv] Saved to: {args.out_csv}")

    # Print the best index and interpolation method, if at least one test succeeded.
    if len(df) > 0:
        best = df.iloc[0]

        print("\n[best result]")
        print(
            f"Best raw index = {int(best['raw_index'])}, "
            f"interpolation = {best['interp']}, "
            f"r_support = {best['r_support']:.5f}, "
            f"nrmse = {best['nrmse']:.5f}"
        )


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()
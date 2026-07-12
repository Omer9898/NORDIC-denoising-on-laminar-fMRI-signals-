#!/usr/bin/env python3
"""
07_fix_nordic_headers.py
------------------------

Fix qform and sform metadata codes for post-NORDIC 4D magnitude images.

For each run, the script:

1. Finds a magnitude NIfTI file containing "mag" in its filename.
2. Loads the post-NORDIC 4D image.
3. Loads a pre-NORDIC reference image.
4. Keeps the post-NORDIC image data and affine matrix unchanged.
5. Copies only qform_code and sform_code from the pre-NORDIC reference.
6. Saves a corrected post-NORDIC 4D image.
7. Reloads the output to verify shape and affine preservation.

Important:
This script does NOT resample, reorient, register, or modify image intensities.
It only updates NIfTI header metadata codes.
"""

# Import os for file and directory operations.
import os

# Import NumPy for affine-matrix comparison and array handling.
import numpy as np

# Import NiBabel for loading, editing, and saving NIfTI images.
import nibabel as nib


# --------------------------------------------------------------------------- #
# Input/output paths
# --------------------------------------------------------------------------- #

# Output directory for corrected NORDIC 4D images.
BASE_DIR = r"D:/NORDIC_Subject1_project/GLM after nordic"

# Pre-NORDIC reference image used only to obtain qform_code and sform_code.
PRE_REF = (
    r"D:/NORDIC_Subject1_project/NEW Regstration files/"
    r"warp_outputs/pre_vol0000.nii.gz"
)

# Dictionary defining each run number and its NORDIC output folder.
RUN_FOLDERS = {
    1: (
        r"D:/FINAL RUN 1/OY_subject1_run1+run2+run3/"
        r"OY - Files after Nordic/output_nordic_1"
    ),
    2: (
        r"D:/FINAL RUN 1/OY_subject1_run1+run2+run3/"
        r"OY - Files after Nordic/output_run_2"
    ),
    3: (
        r"D:/FINAL RUN 1/OY_subject1_run1+run2+run3/"
        r"OY - Files after Nordic/output_run_3"
    ),
}


# --------------------------------------------------------------------------- #
# Find the magnitude NIfTI file
# --------------------------------------------------------------------------- #

def find_mag_file(folder):
    """
    Find a compressed NIfTI magnitude image in a directory.

    A valid file must:
    - End with '.nii.gz'
    - Contain the text 'mag' in its filename

    If multiple candidates are found, the first file in alphabetical order
    is selected.

    Parameters
    ----------
    folder : str
        Directory containing NORDIC output files.

    Returns
    -------
    str
        Full path to the selected magnitude NIfTI file.
    """

    # Store all matching NIfTI magnitude files.
    candidates = []

    # Inspect every file in the specified folder.
    for name in os.listdir(folder):

        # Convert the filename to lowercase for case-insensitive matching.
        low = name.lower()

        # Keep compressed NIfTI files containing "mag" in the filename.
        if low.endswith(".nii.gz") and "mag" in low:
            candidates.append(os.path.join(folder, name))

    # Stop if no magnitude NIfTI file was found.
    if not candidates:
        raise FileNotFoundError(
            f"No magnitude '.nii.gz' file containing 'mag' was found in: {folder}"
        )

    # Sort candidates to ensure consistent selection.
    candidates.sort()

    # Return the first candidate.
    return candidates[0]


# --------------------------------------------------------------------------- #
# Correct NIfTI qform/sform codes for one run
# --------------------------------------------------------------------------- #

def fix_header_one_run(run_id, run_folder, pre_ref, out_dir):
    """
    Correct qform_code and sform_code for one post-NORDIC run.

    The image data, affine matrix, dimensions, voxel sizes, and datatype are
    preserved from the post-NORDIC input image. Only the qform and sform codes
    are copied from the pre-NORDIC reference image.

    Parameters
    ----------
    run_id : int
        Identifier for the functional run.
    run_folder : str
        Folder containing NORDIC output files for this run.
    pre_ref : str
        Path to the pre-NORDIC reference NIfTI image.
    out_dir : str
        Directory where the corrected 4D NIfTI image will be saved.
    """

    # Skip this run if its input directory does not exist.
    if not os.path.exists(run_folder):
        print(f"[skip] Run {run_id}: folder not found -> {run_folder}")
        return

    # Find the NORDIC magnitude image for this run.
    try:
        input_post_4d = find_mag_file(run_folder)

    # Skip the run if an appropriate magnitude file cannot be found.
    except Exception as e:
        print(f"[skip] Run {run_id}: {e}")
        return

    # Define the output filename for the corrected NORDIC 4D image.
    output_fixed_4d = os.path.join(
        out_dir,
        f"NORDIC_run{run_id}_mag_fixhdr.nii.gz"
    )

    # Load the post-NORDIC 4D image.
    img = nib.load(input_post_4d)

    # Load the pre-NORDIC reference image.
    ref = nib.load(pre_ref)

    # Copy the post-NORDIC affine matrix.
    # This keeps the post-NORDIC image geometry unchanged.
    aff = img.affine.copy()

    # Read qform and sform codes from the reference image.
    qcode = int(ref.header["qform_code"])
    scode = int(ref.header["sform_code"])

    # Warn if input and reference affines differ.
    # The output still retains the input image affine, not the reference affine.
    if not np.allclose(img.affine, ref.affine, atol=1e-4):
        print(
            f"[warn] Run {run_id}: input and PRE_REF affines differ. "
            "The input affine will be retained; only qform/sform codes are copied."
        )

    # Create a new NIfTI image using:
    # - Original post-NORDIC voxel data
    # - Original post-NORDIC affine matrix
    # - A copy of the original post-NORDIC header
    new = nib.Nifti1Image(
        np.asanyarray(img.dataobj),
        aff,
        header=img.header.copy()
    )

    # Set qform matrix to the retained input affine and assign reference qform code.
    new.set_qform(aff, code=qcode)

    # Set sform matrix to the retained input affine and assign reference sform code.
    new.set_sform(aff, code=scode)

    # Preserve the original post-NORDIC image data type.
    new.set_data_dtype(img.get_data_dtype())

    # Save the corrected NIfTI file.
    nib.save(new, output_fixed_4d)

    # Reload the output image to confirm successful writing.
    chk = nib.load(output_fixed_4d)

    # Print validation results.
    print(f"\n[run {run_id}]")
    print(f"Input file : {input_post_4d}")
    print(f"Output file: {output_fixed_4d}")

    # Print input qform and sform codes.
    print(
        f"Input qform/sform codes : "
        f"{int(img.header['qform_code'])}/{int(img.header['sform_code'])}"
    )

    # Print output qform and sform codes.
    print(
        f"Output qform/sform codes: "
        f"{int(chk.header['qform_code'])}/{int(chk.header['sform_code'])}"
    )

    # Confirm that the affine matrix was retained.
    print(
        f"Affine preserved: "
        f"{np.allclose(chk.affine, aff, atol=1e-6)}"
    )

    # Confirm that the full 4D image shape was retained.
    print(f"Shape preserved : {chk.shape == img.shape}")


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #

def main():
    """
    Process all defined NORDIC functional runs and correct NIfTI header codes.
    """

    # Confirm that the pre-NORDIC reference image exists.
    if not os.path.exists(PRE_REF):
        raise FileNotFoundError(f"PRE_REF not found: {PRE_REF}")

    # Create the output directory if it does not already exist.
    os.makedirs(BASE_DIR, exist_ok=True)

    # Print workflow information.
    print("[start]")
    print(f"Output directory: {BASE_DIR}")
    print(f"Pre-NORDIC reference: {PRE_REF}")

    # Process each run defined in RUN_FOLDERS.
    for run_id, run_folder in RUN_FOLDERS.items():
        fix_header_one_run(
            run_id,
            run_folder,
            PRE_REF,
            BASE_DIR
        )

    # Confirm completion.
    print("\n[done] All available runs were processed.")


# Execute the workflow only when the script is run directly.
if __name__ == "__main__":
    main()
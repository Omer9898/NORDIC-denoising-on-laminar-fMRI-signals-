#!/usr/bin/env python3
"""
01_extract_volume.py

Extracts one 3D volume from a 4D functional NIfTI file
while preserving its spatial geometry and metadata.
"""

# Import modules for command-line arguments, file handling, numerical arrays,
# and reading/writing NIfTI medical-image files.
import argparse
import os
import numpy as np
import nibabel as nib


# Default directory where the extracted 3D volume will be saved.
OUTPUT_DIR = r"D:/NORDIC_Subject1_project/NEW Regstration files"


def extract_volume(func_path: str, index: int, out_dir: str) -> None:
    """
    Extract one 3D volume from a 4D functional NIfTI image.

    Parameters
    ----------
    func_path : str
        Path to the input 4D NIfTI file.
    index : int
        Zero-based index of the volume to extract.
        For example, index 0 extracts the first volume.
    out_dir : str
        Directory where the extracted 3D NIfTI file will be saved.
    """

    # Check whether the input NIfTI file exists.
    if not os.path.isfile(func_path):
        raise FileNotFoundError(f"Input file not found: {func_path}")

    # Create the output directory if it does not already exist.
    os.makedirs(out_dir, exist_ok=True)

    # Define the output filename using the selected volume index.
    out_path = os.path.join(out_dir, f"mag_POCS_r1_vol{index:04d}.nii.gz")

    # Load the original functional NIfTI image.
    img = nib.load(func_path)

    # Confirm that the input image is 4D: X, Y, Z, and time/volume dimension.
    if img.ndim != 4:
        raise ValueError(
            f"Expected a 4D image, but got an image with shape {img.shape}"
        )

    # Get the total number of 3D volumes in the fourth dimension.
    n_t = img.shape[3]

    # Ensure that the requested volume index is valid.
    if not (0 <= index < n_t):
        raise IndexError(
            f"Index {index} is out of range for {n_t} volumes "
            f"(valid indices: 0 to {n_t - 1})"
        )

    # Extract the selected 3D volume without loading unnecessary volumes.
    vol = np.asarray(img.dataobj[..., index])

    # Copy the original header to preserve relevant NIfTI metadata.
    hdr = img.header.copy()

    # Update the header shape from 4D to the extracted 3D volume shape.
    hdr.set_data_shape(vol.shape)

    # Preserve the qform spatial transformation matrix and its code.
    hdr.set_qform(img.get_qform(), code=int(img.header["qform_code"]))

    # Preserve the sform spatial transformation matrix and its code.
    hdr.set_sform(img.get_sform(), code=int(img.header["sform_code"]))

    # Preserve voxel dimensions in the X, Y, and Z directions.
    hdr.set_zooms(img.header.get_zooms()[:3])

    # Create a new 3D NIfTI image using the extracted volume,
    # original affine matrix, and copied/updated header.
    out = nib.Nifti1Image(vol, img.affine, header=hdr)

    # Preserve the original image data type, for example int16 or float32.
    out.set_data_dtype(img.get_data_dtype())

    # Save the extracted 3D NIfTI image to disk.
    nib.save(out, out_path)

    # Reload the saved file to verify that it was written correctly.
    chk = nib.load(out_path)

    # Print validation information in the terminal.
    print(f"[extract] Output file: {out_path}")
    print(f"[extract] Output shape: {chk.shape}")
    print(
        f"[extract] Affine preserved: "
        f"{np.allclose(chk.affine, img.affine)}"
    )
    print(
        f"[extract] Voxel sizes preserved: "
        f"{np.allclose(chk.header.get_zooms()[:3], img.header.get_zooms()[:3])}"
    )


if __name__ == "__main__":
    # Create a command-line argument parser.
    ap = argparse.ArgumentParser(
        description="Extract one 3D volume from a 4D functional NIfTI file."
    )

    # Input path of the 4D functional NIfTI image.
    ap.add_argument(
        "--func",
        default=r"D:\Downloads\run1\func\mag_POCS_r1.nii.gz",
        help="Path to the input 4D functional NIfTI file."
    )

    # Index of the 3D volume to extract; 0 means the first volume.
    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index of the 3D volume to extract."
    )

    # Directory in which the extracted 3D image will be saved.
    ap.add_argument(
        "--outdir",
        default=OUTPUT_DIR,
        help="Output directory for the extracted NIfTI volume."
    )

    # Read the user-provided command-line arguments.
    args = ap.parse_args()

    # Run the extraction function.
    extract_volume(args.func, args.index, args.outdir)

# #!/usr/bin/env python3
# """
# 01_extract_volume.py
# --------------------
# Extract ONE 3D volume from a 4D functional NIfTI, preserving geometry exactly.

# Why this is non-trivial:
# - The 3D volume must keep the SAME 3x3 affine + origin as the 4D file
#   (same qform/sform, same zooms). We copy the header and only drop the
#   time dimension. We do NOT let nibabel recompute anything.
# - We also copy dtype and scl_slope/scl_inter so intensities are identical.

# Usage:
#     python 01_extract_volume.py \
#         --func   /path/mag_POCS_r1.nii.gz \
#         --index  0 \
#         --out    /path/mag_POCS_r1_vol0000.nii.gz

# Note on indexing: --index is the 0-based RAW time index into the 4D array.
# See the README / 02_reproduce_warp.py --scan-indices for resolving the
# "1000" filename label to a raw index empirically.
# """
# import argparse
# import numpy as np
# import nibabel as nib


# def extract_volume(func_path: str, index: int, out_path: str) -> None:
#     img = nib.load(func_path)
#     if img.ndim != 4:
#         raise ValueError(f"Expected a 4D image, got shape {img.shape}")
#     n_t = img.shape[3]
#     if not (0 <= index < n_t):
#         raise IndexError(f"index {index} out of range for {n_t} volumes (0..{n_t-1})")

#     # dataobj slicing avoids loading the whole 4D array into memory.
#     vol = np.asarray(img.dataobj[..., index])

#     # Preserve geometry: reuse the source header, only fix dim/pixdim for 3D.
#     hdr = img.header.copy()
#     hdr.set_data_shape(vol.shape)          # sets dim[0]=3, dim[4]=1
#     # keep qform/sform codes + affines exactly as in the 4D file
#     hdr.set_qform(img.get_qform(), code=int(img.header["qform_code"]))
#     hdr.set_sform(img.get_sform(), code=int(img.header["sform_code"]))
#     # zero the temporal pixdim so it doesn't leak into the 3D file
#     zooms = list(img.header.get_zooms()[:3])
#     hdr.set_zooms(zooms)

#     out = nib.Nifti1Image(vol, img.affine, header=hdr)
#     # match the on-disk dtype of the source
#     out.set_data_dtype(img.get_data_dtype())
#     nib.save(out, out_path)

#     # ---- verification printout ----
#     chk = nib.load(out_path)
#     print(f"[extract] source 4D shape  : {img.shape}")
#     print(f"[extract] extracted index  : {index} (0-based)")
#     print(f"[extract] output shape     : {chk.shape}")
#     print(f"[extract] zooms preserved  : {np.allclose(chk.header.get_zooms()[:3], zooms)}")
#     print(f"[extract] affine preserved : {np.allclose(chk.affine, img.affine)}")
#     print(f"[extract] qform/sform codes: {int(chk.header['qform_code'])}/{int(chk.header['sform_code'])}")
#     print(f"[extract] saved -> {out_path}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--func", required=True, help="4D functional .nii.gz")
#     ap.add_argument("--index", type=int, required=True, help="0-based raw time index")
#     ap.add_argument("--out", required=True, help="output 3D .nii.gz")
#     a = ap.parse_args()
#     extract_volume(a.func, a.index, a.out)

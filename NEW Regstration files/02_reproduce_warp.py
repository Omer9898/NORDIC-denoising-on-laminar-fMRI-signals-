#!/usr/bin/env python3
"""
02_apply_transform_chain.py

Applies a sequence of spatial transformations to an extracted 3D NIfTI volume
using ANTsPy, then optionally compares the reproduced output with a known
ground-truth image.

Typical transformation chain:
1. Registration from functional image space to anatomical space
2. Distortion-correction nonlinear warp
3. Distortion-correction affine transformation
4. Optional motion-correction transformation
"""

# Import modules for command-line arguments, file management, numerical analysis,
# NIfTI image handling, and ANTs image registration/transformation tools.
import argparse
import os
import numpy as np
import nibabel as nib
import ants


# Default output directory for transformed images.
OUTPUT_DIR = "D:/NORDIC_Subject1_project/NEW Regstration files"


def build_chain(reg2anat, warp_distcorr, affine_distcorr, motion=None, order="standard"):
    """
    Build the transformation list used by ANTs apply_transforms.

    Parameters
    ----------
    reg2anat : str
        Transform file that maps the image from functional space to anatomical space.
    warp_distcorr : str
        Nonlinear deformation field used for distortion correction.
    affine_distcorr : str
        Affine transform used for distortion correction.
    motion : str or None
        Optional motion-correction transform for a specific volume.
    order : str
        Defines the order of transformations in the chain.

        Available options:
        - "standard": registration-to-anatomy, nonlinear warp, affine correction
        - "affine_before_warp": registration-to-anatomy, affine correction, nonlinear warp
        - "distcorr_only": nonlinear warp and affine correction only

    Returns
    -------
    list
        Ordered list of transformation files.
    """

    # Standard transformation order:
    # anatomical registration -> distortion warp -> distortion affine transform.
    if order == "standard":
        chain = [reg2anat, warp_distcorr, affine_distcorr]

    # Alternative order where the affine distortion correction comes before the warp.
    elif order == "affine_before_warp":
        chain = [reg2anat, affine_distcorr, warp_distcorr]

    # Apply only distortion-correction transforms without functional-to-anatomical registration.
    elif order == "distcorr_only":
        chain = [warp_distcorr, affine_distcorr]

    # Raise an error if an unsupported transformation order is selected.
    else:
        raise ValueError(f"Unknown transformation order: '{order}'")

    # Add the motion transform to the end of the chain if one was provided.
    if motion:
        chain = chain + [motion]

    # Return the complete transformation chain.
    return chain


def apply_chain(moving_path, reference_path, chain, out_path, interp="lanczosWindowedSinc"):
    """
    Apply a transformation chain to a moving image using ANTsPy.

    Parameters
    ----------
    moving_path : str
        Path to the input image that will be transformed.
    reference_path : str
        Path to the fixed/reference image that defines the output space,
        image dimensions, voxel grid, and orientation.
    chain : list
        Ordered list of ANTs transformation files.
    out_path : str
        Path where the transformed output image will be saved.
    interp : str
        Interpolation method used during resampling.
        Default: "lanczosWindowedSinc", which is suitable for high-quality
        interpolation of continuous intensity images.

    Returns
    -------
    str
        Path to the saved transformed image.
    """

    # Load the image to be transformed.
    mov = ants.image_read(moving_path)

    # Load the reference image that defines the target anatomical space.
    ref = ants.image_read(reference_path)

    # Apply all transformations in the specified order.
    out = ants.apply_transforms(
        fixed=ref,
        moving=mov,
        transformlist=chain,
        interpolator=interp
    )

    # Save the transformed image as a NIfTI file.
    ants.image_write(out, out_path)

    # Return the output file path.
    return out_path


def compare_images(reproduced_path, truth_path):
    """
    Compare a reproduced transformed image with a ground-truth image.

    The comparison includes:
    - Image shape consistency
    - Affine matrix consistency
    - Pearson correlation coefficient
    - Root Mean Squared Error (RMSE)
    - Normalized RMSE (NRMSE)

    Parameters
    ----------
    reproduced_path : str
        Path to the transformed image created by this script.
    truth_path : str
        Path to the expected or ground-truth transformed image.

    Returns
    -------
    dict
        Dictionary containing comparison metrics.
    """

    # Load the reproduced image and the ground-truth image.
    a = nib.load(reproduced_path)
    b = nib.load(truth_path)

    # Verify that both images have the same dimensions.
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: reproduced image {a.shape} vs ground truth {b.shape}"
        )

    # Check whether the spatial affine matrices are approximately identical.
    if not np.allclose(a.affine, b.affine, atol=1e-3):
        print("[compare] WARNING: Affine matrices differ.")

    # Convert both images to float64 arrays and flatten them into one-dimensional vectors.
    x = a.get_fdata(dtype=np.float64).ravel()
    y = b.get_fdata(dtype=np.float64).ravel()

    # Keep only voxels where both images contain finite values.
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    # Define the comparison support as voxels where at least one image is non-zero.
    # This excludes shared background voxels with zero intensity.
    supp = (x != 0) | (y != 0)
    xs = x[supp]
    ys = y[supp]

    # Stop if no valid non-background voxels are available for comparison.
    if xs.size == 0:
        raise ValueError("No valid non-zero voxels available for comparison.")

    # Calculate Pearson correlation between voxel intensities.
    r = float(np.corrcoef(xs, ys)[0, 1])

    # Calculate the Root Mean Squared Error between the two images.
    rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))

    # Normalize RMSE by the intensity range of the ground-truth image.
    # A small value is added to avoid division by zero.
    nrmse = rmse / (ys.max() - ys.min() + 1e-12)

    # Return all calculated metrics.
    return {
        "pearson_r": r,
        "rmse": rmse,
        "nrmse": nrmse,
        "n_support_voxels": int(xs.size)
    }


def print_metrics(tag, m):
    """
    Print image-comparison metrics in a readable format.

    Parameters
    ----------
    tag : str
        Label describing the tested transformation order.
    m : dict
        Dictionary containing correlation and error metrics.
    """

    # Print the name of the tested transformation chain.
    print(f"[compare] {tag}")

    # Print Pearson correlation; values closer to 1 indicate stronger similarity.
    print(f"  Pearson r : {m['pearson_r']:.6f}")

    # Print RMSE; lower values indicate smaller intensity differences.
    print(f"  RMSE      : {m['rmse']:.6f}")

    # Print normalized RMSE; lower values indicate better agreement.
    print(f"  NRMSE     : {m['nrmse']:.6f}")

    # Print the number of non-background voxels used in the comparison.
    print(f"  Support   : {m['n_support_voxels']}")


def main():
    """
    Parse command-line arguments, construct a transformation chain,
    apply it to the input image, and optionally compare the result
    with a ground-truth warped image.
    """

    # Create a command-line argument parser.
    ap = argparse.ArgumentParser(
        description=(
            "Apply a spatial transformation chain to a 3D NIfTI volume "
            "and optionally compare it with a ground-truth image."
        )
    )

    # Input 3D functional volume that will be transformed.
    ap.add_argument(
        "--moving",
        default=f"{OUTPUT_DIR}/mag_POCS_r1_vol0000.nii.gz",
        help="Path to the extracted 3D moving volume."
    )

    # Reference image defining the target anatomical-space grid.
    ap.add_argument(
        "--reference",
        default="D:/Downloads/run1/func/mag_POCS_r1_1000_Warped-to-Anat.nii.gz",
        help="Path to the reference image in anatomical space."
    )

    # Affine transform generated during distortion correction.
    ap.add_argument(
        "--affine-distcorr",
        default="D:/Downloads/run1/func/mag_POCS_r1_DistCorr_00GenericAffine.mat",
        help="Path to the affine distortion-correction transform."
    )

    # Nonlinear warp field generated during distortion correction.
    ap.add_argument(
        "--warp-distcorr",
        default="D:/Downloads/run1/func/mag_POCS_r1_DistCorr_01Warp.nii.gz",
        help="Path to the nonlinear distortion-correction warp field."
    )

    # Registration transform mapping the functional image to anatomical space.
    ap.add_argument(
        "--reg2anat",
        default="D:/Downloads/run1/func/custom_reg2anat.txt",
        help="Path to the functional-to-anatomical registration transform."
    )

    # Optional volume-specific motion-correction transform.
    ap.add_argument(
        "--motion",
        default=None,
        help="Optional motion-correction transform."
    )

    # Output path for the reproduced transformed image.
    ap.add_argument(
        "--out",
        default=f"{OUTPUT_DIR}/mag_POCS_r1_vol0000_reproduced_Warped-to-Anat.nii.gz",
        help="Path for the saved transformed output image."
    )

    # Optional ground-truth image used to assess the accuracy of the result.
    ap.add_argument(
        "--compare",
        default="D:/Downloads/run1/func/mag_POCS_r1_1000_Warped-to-Anat.nii.gz",
        help="Path to the ground-truth image used for comparison."
    )

    # Interpolation method used while resampling the image.
    ap.add_argument(
        "--interp",
        default="lanczosWindowedSinc",
        help="Interpolation method, for example linear or lanczosWindowedSinc."
    )

    # Optional flag to test additional transformation-order variants.
    ap.add_argument(
        "--try-variants",
        action="store_true",
        help="Test alternative orders of distortion-correction transforms."
    )

    # Read arguments supplied in the command line or use defaults.
    args = ap.parse_args()

    # Ensure that the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build the default transformation chain.
    chain = build_chain(
        args.reg2anat,
        args.warp_distcorr,
        args.affine_distcorr,
        args.motion,
        order="standard"
    )

    # Display the transforms that will be applied.
    print("[chain] Transform list:")
    for t in chain:
        print(" -", t)

    # Apply the standard transformation chain to the moving image.
    apply_chain(
        args.moving,
        args.reference,
        chain,
        args.out,
        args.interp
    )

    # Confirm where the transformed image was saved.
    print(f"[apply] Saved output: {args.out}")

    # Compare the output with the ground truth if a comparison image was provided.
    if args.compare:
        m = compare_images(args.out, args.compare)
        print_metrics("Standard order", m)

    # Optionally test other possible transform orders.
    if args.try_variants:
        print("\n[variants] Trying alternative transformation orders...")

        # Test the alternative transform orders defined in build_chain().
        for name in ["affine_before_warp", "distcorr_only"]:

            # Generate a unique output filename for each tested variant.
            tmp_out = args.out.replace(".nii.gz", f".{name}.nii.gz")

            # Build the transformation chain for the current variant.
            v_chain = build_chain(
                args.reg2anat,
                args.warp_distcorr,
                args.affine_distcorr,
                args.motion,
                order=name
            )

            # Apply the variant transformation chain.
            apply_chain(
                args.moving,
                args.reference,
                v_chain,
                tmp_out,
                args.interp
            )

            # Print the output location for this variant.
            print(f"[apply] Variant saved: {tmp_out}")

            # Compare the variant output with the ground truth if available.
            if args.compare:
                vm = compare_images(tmp_out, args.compare)
                print_metrics(name, vm)


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()

    
# import argparse
# import os
# import tempfile
# import numpy as np
# import nibabel as nib
# import ants


# # ----------------------------------------------------------------------------- 
# # core: build the transformlist in ANTs order (reference/anat-side first)
# # -----------------------------------------------------------------------------
# def build_chain(reg2anat, warp_distcorr, affine_distcorr, motion=None, order="standard"):
#     if order == "standard":
#         chain = [reg2anat, warp_distcorr, affine_distcorr]
#     elif order == "affine_before_warp":
#         chain = [reg2anat, affine_distcorr, warp_distcorr]
#     elif order == "distcorr_only":
#         chain = [warp_distcorr, affine_distcorr]  # no reg2anat (sanity variant)
#     else:
#         raise ValueError(f"unknown order '{order}'")
#     if motion:
#         chain = chain + [motion]   # motion is closest to the moving image -> last
#     return chain


# def apply_chain(moving_path, reference_path, chain, out_path, interp="linear"):
#     mov = ants.image_read(moving_path)
#     ref = ants.image_read(reference_path)
#     out = ants.apply_transforms(fixed=ref, moving=mov,
#                                 transformlist=chain, interpolator=interp)
#     ants.image_write(out, out_path)
#     return out_path


# # ----------------------------------------------------------------------------- 
# # comparison metrics
# # -----------------------------------------------------------------------------
# def compare(reproduced_path, truth_path):
#     a = nib.load(reproduced_path)
#     b = nib.load(truth_path)
#     if a.shape != b.shape:
#         raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
#     if not np.allclose(a.affine, b.affine, atol=1e-3):
#         print("[compare] WARNING: affines differ; are you on the same reference grid?")
#     x = a.get_fdata(dtype=np.float64).ravel()
#     y = b.get_fdata(dtype=np.float64).ravel()
#     m = np.isfinite(x) & np.isfinite(y)
#     x, y = x[m], y[m]
#     # restrict to the union of nonzero support so background zeros don't inflate r
#     supp = (x != 0) | (y != 0)
#     xs, ys = x[supp], y[supp]
#     r = float(np.corrcoef(xs, ys)[0, 1])
#     rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))
#     nrmse = rmse / (ys.max() - ys.min() + 1e-12)
#     return {"pearson_r": r, "rmse": rmse, "nrmse": nrmse,
#             "n_support_voxels": int(supp.sum())}


# def _print_metrics(tag, m):
#     print(f"[compare] {tag:<28} r={m['pearson_r']:.6f}  "
#           f"RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}  "
#           f"(support={m['n_support_voxels']})")


# # ----------------------------------------------------------------------------- 
# def main():
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--moving", help="extracted 3D volume (from 01_extract_volume.py)")
#     ap.add_argument("--reference", required=True,
#                     help="reference grid = an anat-space file (defines 0.75mm grid)")
#     ap.add_argument("--affine-distcorr", required=True)
#     ap.add_argument("--warp-distcorr", required=True)
#     ap.add_argument("--reg2anat", required=True)
#     ap.add_argument("--motion", default=None, help="optional per-volume motion .mat")
#     ap.add_argument("--out", help="output reproduced anat-space volume")
#     ap.add_argument("--interp", default="linear",
#                     help="linear | nearestNeighbor | lanczosWindowedSinc | bSpline")
#     ap.add_argument("--compare", help="ground-truth *_Warped-to-Anat.nii.gz to score against")
#     ap.add_argument("--try-variants", action="store_true",
#                     help="also test alternative chain orderings and report corr/RMSE")
#     ap.add_argument("--func", help="4D functional (needed for --scan-indices)")
#     ap.add_argument("--scan-indices",
#                     help="comma-separated 0-based raw indices to test against --compare")
#     a = ap.parse_args()

#     # ---- mode A: scan candidate raw indices to resolve the filename label ----
#     if a.scan_indices:
#         if not (a.func and a.compare):
#             ap.error("--scan-indices requires --func and --compare")
#         from importlib import import_module  # reuse the extractor
#         ex = import_module("01_extract_volume") if os.path.exists("01_extract_volume.py") \
#             else None
#         idxs = [int(s) for s in a.scan_indices.split(",")]
#         img = nib.load(a.func)
#         print(f"[scan] 4D shape {img.shape}; testing indices {idxs} against {os.path.basename(a.compare)}")
#         results = []
#         with tempfile.TemporaryDirectory() as td:
#             for idx in idxs:
#                 if not (0 <= idx < img.shape[3]):
#                     print(f"[scan] index {idx} out of range -> skipped")
#                     continue
#                 mv = os.path.join(td, f"vol{idx:04d}.nii.gz")
#                 vol = np.asarray(img.dataobj[..., idx])
#                 hdr = img.header.copy(); hdr.set_data_shape(vol.shape)
#                 nib.save(nib.Nifti1Image(vol, img.affine, header=hdr), mv)
#                 chain = build_chain(a.reg2anat, a.warp_distcorr, a.affine_distcorr,
#                                     a.motion, "standard")
#                 rep = os.path.join(td, f"rep{idx:04d}.nii.gz")
#                 apply_chain(mv, a.reference, chain, rep, a.interp)
#                 m = compare(rep, a.compare)
#                 _print_metrics(f"raw index {idx}", m)
#                 results.append((idx, m["pearson_r"]))
#         if results:
#             best = max(results, key=lambda t: t[1])
#             print(f"\n[scan] BEST MATCH -> raw index {best[0]} (r={best[1]:.6f})")
#             print(f"[scan] Interpretation: filename label maps to this 0-based raw index.")
#         return

#     # ---- mode B: single reproduction (+ optional compare / variants) ----
#     if not (a.moving and a.out):
#         ap.error("single-reproduction mode needs --moving and --out")

#     chain = build_chain(a.reg2anat, a.warp_distcorr, a.affine_distcorr, a.motion, "standard")
#     print("[chain] transformlist (ANTs order, reference-side first):")
#     for t in chain:
#         print("        -t", t)
#     apply_chain(a.moving, a.reference, chain, a.out, a.interp)
#     print(f"[apply] interpolator = {a.interp}")
#     print(f"[apply] saved -> {a.out}")

#     if a.compare:
#         _print_metrics("standard order", compare(a.out, a.compare))

#     if a.try_variants:
#         print("\n[variants] testing alternative orderings:")
#         for name in ["affine_before_warp", "distcorr_only"]:
#             v_chain = build_chain(a.reg2anat, a.warp_distcorr, a.affine_distcorr,
#                                   a.motion, name)
#             tmp = a.out.replace(".nii.gz", f".{name}.nii.gz")
#             apply_chain(a.moving, a.reference, v_chain, tmp, a.interp)
#             if a.compare:
#                 _print_metrics(name, compare(tmp, a.compare))


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
06_postnordic_native_glm.py
---------------------------

Run a first-level GLM on post-NORDIC native-space 4D functional MRI data.

This script:
1. Loads a post-NORDIC 4D functional NIfTI image.
2. Loads an events table containing onset, duration, and trial_type columns.
3. Optionally loads nuisance confounds and a native-space mask.
4. Fits a voxel-wise first-level GLM using Nilearn.
5. Computes requested contrast maps.
6. Saves effect-size, z-score, and t-statistic images.
7. Saves the design matrix and estimates median temporal SNR.

Important:
- The functional image remains in native space.
- The optional mask must be in the same native space as the 4D image.
- If anatomical-space maps are needed later, warp the resulting effect/z/t maps,
  not the entire 4D functional time series.
"""

# Import argparse for command-line argument handling.
import argparse

# Import os for path validation and directory creation.
import os

# Import NumPy for numerical calculations.
import numpy as np

# Import pandas for reading event/confound tables and saving the design matrix.
import pandas as pd

# Import NiBabel for reading and saving NIfTI files.
import nibabel as nib

# Import Nilearn's voxel-wise first-level GLM model.
from nilearn.glm.first_level import FirstLevelModel


# --------------------------------------------------------------------------- #
# Default input and output paths
# --------------------------------------------------------------------------- #

# Path to the post-NORDIC native-space 4D functional image.
FUNC = r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run1_mag_fixhdr.nii.gz"

# Directory where GLM maps, design matrix, and copied events file are saved.
OUT_DIR = r"D:/NORDIC_Subject1_project/GLM after nordic/glm_postnordic_run1"

# Default repetition time in seconds.
TR_DEFAULT = 3.0


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_table(path):
    """
    Read a CSV or TSV table automatically.

    Parameters
    ----------
    path : str
        Path to a CSV or TSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """

    # Detect the delimiter automatically, supporting CSV and TSV input.
    return pd.read_csv(path, sep=None, engine="python")


def tsnr(func_path, mask_path=None):
    """
    Calculate the median temporal signal-to-noise ratio (tSNR).

    Temporal SNR is calculated voxel-wise as:

        mean signal across time / standard deviation across time

    Parameters
    ----------
    func_path : str
        Path to the 4D functional NIfTI image.
    mask_path : str or None
        Optional binary mask in the same native space as the functional image.

    Returns
    -------
    float
        Median voxel-wise tSNR within the mask, or within an automatically
        selected non-background region if no mask is provided.
    """

    # Load the 4D functional image.
    img = nib.load(func_path)

    # Load image data as float32 to reduce memory usage.
    d = img.get_fdata(dtype=np.float32)

    # Compute the mean signal across the time dimension.
    mean = d.mean(axis=-1)

    # Compute temporal standard deviation across the time dimension.
    sd = d.std(axis=-1)

    # Calculate tSNR voxel-wise while avoiding division by zero.
    t = np.divide(
        mean,
        sd,
        out=np.zeros_like(mean),
        where=sd > 0
    )

    # If a mask is provided, calculate median tSNR inside the mask only.
    if mask_path:
        m = nib.load(mask_path).get_fdata() > 0

        # Verify that mask and functional spatial dimensions agree.
        if m.shape != t.shape:
            raise ValueError(
                f"Mask shape {m.shape} does not match functional image shape {t.shape}"
            )

        # Return median tSNR inside the binary mask.
        return float(np.median(t[m]))

    # Without a mask, use voxels whose mean signal is above the global mean.
    # This acts as a simple approximation to exclude most zero background.
    return float(np.median(t[mean > mean.mean()]))


def parse_contrasts(spec):
    """
    Convert a comma-separated contrast specification into a dictionary.

    Examples
    --------
    Input:
        "pain,touch,pain-touch"

    Output:
        {
            "pain": "pain",
            "touch": "touch",
            "pain-touch": "pain - touch"
        }

    Parameters
    ----------
    spec : str
        Comma-separated contrast names or simple A-B expressions.

    Returns
    -------
    dict
        Dictionary mapping output names to Nilearn contrast expressions.
    """

    # Create an empty dictionary for contrast names and expressions.
    out = {}

    # Split the input string by commas and remove empty entries.
    tokens = [s.strip() for s in spec.split(",") if s.strip()]

    # Process each requested contrast.
    for tok in tokens:

        # Interpret expressions such as "pain-touch" as "pain - touch".
        if "-" in tok:
            condition_a, condition_b = [p.strip() for p in tok.split("-", 1)]
            out[tok] = f"{condition_a} - {condition_b}"

        # A single condition is interpreted as condition versus implicit baseline.
        else:
            out[tok] = tok

    # Return contrast definitions.
    return out


def validate_events(events):
    """
    Validate and prepare an events table for Nilearn.

    Required columns:
    - onset
    - duration
    - trial_type

    Parameters
    ----------
    events : pandas.DataFrame
        Input events table.

    Returns
    -------
    pandas.DataFrame
        Cleaned table containing only the required event columns.
    """

    # Required event-table columns for the GLM.
    required_columns = ("onset", "duration", "trial_type")

    # Confirm that all required columns exist.
    for column in required_columns:
        if column not in events.columns:
            raise ValueError(
                f"Events file is missing the required column: '{column}'"
            )

    # Keep only the required columns and create an independent copy.
    events = events[["onset", "duration", "trial_type"]].copy()

    # Ensure that trial labels are strings.
    events["trial_type"] = events["trial_type"].astype(str)

    # Convert onset and duration values to numeric values.
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    events["duration"] = pd.to_numeric(events["duration"], errors="raise")

    # Reject negative durations because they are not valid event durations.
    if (events["duration"] < 0).any():
        raise ValueError("Events table contains one or more negative durations.")

    # Sort events chronologically to make the input easier to inspect.
    return events.sort_values("onset").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main GLM workflow
# --------------------------------------------------------------------------- #

def main():
    """
    Load post-NORDIC data, fit a first-level GLM, save contrast maps,
    save the design matrix, and report native-space tSNR.
    """

    # Create a command-line argument parser.
    ap = argparse.ArgumentParser(
        description="Run a first-level GLM on post-NORDIC native-space 4D fMRI data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Events table containing onset, duration, and trial_type columns.
    ap.add_argument(
        "--events",
        required=True,
        help="Events file (.csv or .tsv) containing onset, duration, and trial_type."
    )

    # Repetition time in seconds.
    ap.add_argument(
        "--tr",
        type=float,
        default=TR_DEFAULT,
        help=f"Repetition time in seconds. Default: {TR_DEFAULT}"
    )

    # Optional nuisance-regressor table.
    ap.add_argument(
        "--confounds",
        default=None,
        help="Optional confounds table; it must have one row per functional volume."
    )

    # Optional native-space mask.
    ap.add_argument(
        "--mask",
        default=None,
        help="Optional binary mask in the same native space as the functional image."
    )

    # Optional custom contrasts.
    ap.add_argument(
        "--contrasts",
        default=None,
        help=(
            "Comma-separated contrasts, for example: "
            "'pain,touch,pain-touch'. "
            "If omitted, every condition is tested versus baseline."
        )
    )

    # Optional override for functional image path.
    ap.add_argument(
        "--func",
        default=FUNC,
        help="Path to the post-NORDIC native-space 4D functional NIfTI image."
    )

    # Optional override for the output directory.
    ap.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help="Directory where GLM outputs will be saved."
    )

    # Parse the command-line arguments.
    args = ap.parse_args()

    # Verify that the functional image exists.
    if not os.path.exists(args.func):
        raise FileNotFoundError(f"Functional image not found: {args.func}")

    # Verify that the events file exists.
    if not os.path.exists(args.events):
        raise FileNotFoundError(f"Events file not found: {args.events}")

    # Verify that TR is valid.
    if args.tr <= 0:
        raise ValueError(f"TR must be greater than zero, but received {args.tr}")

    # Create the output directory if necessary.
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the functional NIfTI image.
    img = nib.load(args.func)

    # Confirm that the functional image is 4D.
    if img.ndim != 4:
        raise ValueError(
            f"Functional image must be 4D, but its shape is {img.shape}"
        )

    # Get the number of time points/scans.
    n_scans = img.shape[3]

    # Load and validate the event table.
    events = validate_events(read_table(args.events))

    # Check that all events begin within the functional acquisition duration.
    run_duration = n_scans * args.tr
    if (events["onset"] >= run_duration).any():
        raise ValueError(
            f"At least one event onset occurs after the scan duration "
            f"({run_duration:.2f} seconds)."
        )

    # Load optional confounds.
    confounds = None

    if args.confounds:

        # Verify that the confounds file exists.
        if not os.path.exists(args.confounds):
            raise FileNotFoundError(
                f"Confounds file not found: {args.confounds}"
            )

        # Load the confounds table.
        confounds = read_table(args.confounds)

        # The number of rows must match the number of functional volumes.
        if len(confounds) != n_scans:
            raise ValueError(
                f"Confounds rows ({len(confounds)}) do not match "
                f"the number of scans ({n_scans})."
            )

    # Verify that the optional mask exists.
    if args.mask and not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")

    # Configure the first-level fMRI GLM.
    glm = FirstLevelModel(
        # Set the repetition time in seconds.
        t_r=args.tr,

        # Use the Glover hemodynamic response function model.
        hrf_model="glover",

        # Model low-frequency temporal drift using cosine basis functions.
        drift_model="cosine",

        # Remove slow fluctuations with a 128-second high-pass cutoff.
        high_pass=1.0 / 128.0,

        # Do not apply additional spatial smoothing in this script.
        smoothing_fwhm=None,

        # Use an optional native-space mask if one was supplied.
        mask_img=args.mask,

        # Keep information required for later inspection of design matrices.
        minimize_memory=False,

        # Disable automatic signal scaling.
        signal_scaling=0,
    )

    # Print core fitting information.
    print(f"[glm] Functional image: {args.func}")
    print(
        f"[glm] Fitting model: n_scans={n_scans}, "
        f"TR={args.tr}s, "
        f"conditions={sorted(events['trial_type'].unique())}"
    )

    # Fit the GLM using the functional data, events, and optional confounds.
    glm.fit(
        args.func,
        events=events,
        confounds=confounds
    )

    # Identify all experimental conditions from the events table.
    conditions = sorted(events["trial_type"].unique())

    # Use user-defined contrasts if supplied.
    if args.contrasts:
        contrasts = parse_contrasts(args.contrasts)

    # Otherwise, create one contrast per condition versus implicit baseline.
    else:
        contrasts = {condition: condition for condition in conditions}

    # Calculate and save maps for every contrast.
    for name, expression in contrasts.items():

        # Compute the estimated effect-size map.
        effect = glm.compute_contrast(
            expression,
            output_type="effect_size"
        )

        # Compute the z-score map.
        z_map = glm.compute_contrast(
            expression,
            output_type="z_score"
        )

        # Compute the statistic map, usually a t-statistic for this contrast.
        stat_map = glm.compute_contrast(
            expression,
            output_type="stat"
        )

        # Create a safe filename from the contrast name.
        safe_name = (
            name.replace(" ", "")
            .replace("-", "_minus_")
            .replace("+", "_plus_")
        )

        # Save the effect-size image.
        nib.save(
            effect,
            os.path.join(args.out_dir, f"{safe_name}_effect.nii.gz")
        )

        # Save the z-score image.
        nib.save(
            z_map,
            os.path.join(args.out_dir, f"{safe_name}_z.nii.gz")
        )

        # Save the t-statistic image.
        nib.save(
            stat_map,
            os.path.join(args.out_dir, f"{safe_name}_t.nii.gz")
        )

        # Confirm that the contrast maps were created.
        print(
            f"[glm] Contrast '{name}' ({expression}) "
            "-> effect, z, and t maps saved"
        )

    # Save the cleaned events table that was actually used in the model.
    events.to_csv(
        os.path.join(args.out_dir, "events_used.csv"),
        index=False
    )

    # Extract and save the generated design matrix.
    design_matrix = glm.design_matrices_[0]
    design_matrix.to_csv(
        os.path.join(args.out_dir, "design_matrix.csv"),
        index=False
    )

    # Attempt to calculate and report median native-space tSNR.
    try:
        median_tsnr = tsnr(args.func, args.mask)
        print(f"[qc] Native-space median tSNR = {median_tsnr:.2f}")

    # Do not stop GLM completion if tSNR calculation fails.
    except Exception as e:
        print(f"[qc] tSNR calculation skipped: {e}")

    # Print final output information.
    print(f"\n[done] Native-space GLM outputs saved in: {args.out_dir}")
    print(
        "[next] If anatomical-space maps are required, transform the final "
        "*_effect, *_z, or *_t maps. Do not warp the full 4D time series."
    )


# Run the GLM workflow only when this script is executed directly.
if __name__ == "__main__":
    main()
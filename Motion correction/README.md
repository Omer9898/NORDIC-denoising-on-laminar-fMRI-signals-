# motion cor

## Overview
This folder contains motion-correction-related files and supporting material used during preparation of the functional data for post-NORDIC analysis.

## Purpose
The purpose of this section is to store outputs, notes, and quality-control material related to motion handling before first-level GLM analysis.

## Why this step matters
In laminar fMRI, even small spatial inconsistencies can affect cortical-depth specificity. For that reason, motion-related preprocessing should be handled carefully and documented clearly to preserve native-space accuracy as much as possible.

## What belongs in this folder
This folder may include:
- Motion-corrected functional images
- Motion parameter files
- QC plots related to head motion
- Notes describing the motion-correction procedure
- Intermediate files used during motion evaluation

## Relation to the pipeline
The broader post-NORDIC workflow was designed to minimize unnecessary spatial changes, avoid smoothing, and preserve native EPI geometry. Any motion-correction output included here should follow the same principle and support reproducible downstream analysis.

## Notes
Use this folder only for motion-related preprocessing material. Final task-modeling results should be stored in the GLM section.

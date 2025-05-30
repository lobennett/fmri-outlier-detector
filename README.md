# Summary Image Outlier Detection Tool

This tool generates a PDF of summary images and a CSV report to help identify image outliers.  
👉 [View example output files](./survey_medley_output_example/)

### What it does

Given a set of NIfTI images (e.g., contrast estimates across subjects), the tool:

- Generates a PDF with:
  - One page per contrast showing a slice from each subject’s image.
    - Panel titles include user-defined labels (e.g., subject ID, VIF) and the percentage of outlier voxels.
    - Outliers are voxels beyond ±2 standard deviations from the mean at each voxel across subjects.
  - A histogram of outlier percentages across **all** subjects/contrasts.
  - Separate histograms for outlier percentages **per** contrast.
- Saves a CSV with:
  - Subject labels, contrast names, and outlier percentages.
  - Useful for filtering good-quality images based on a chosen threshold.

---

# How to Use

## With `uv`

1. Run `uv sync` to create the `.venv`.
2. Create a script that calls the main function:  
   `generate_all_data_summaries()` from `plotting_functions.py`.  
    * 📎 See an example in [survey_medley_example.py](survey_medley_example.py).  Note this example is a little complicated because I had to extract VIF values from an html because I didn't have a file with the vifs in it (that would have been much easier).
3. Run your script with `uv run`.
   * 📎 See [run_survey_medley_example.batch](./run_survey_medley_example.batch) for an example.

---

## `generate_all_data_summaries()`

This function helps visualize and flag outlier images across multiple contrast maps.

### Inputs

- **data_dictionaries** (`List[Dict[str, Any]]`): Each dictionary should contain:
  - `main_title` (`str`): Title for the group (e.g., contrast name)
  - `nifti_paths` (`List[str]`): Paths to NIfTI images
  - `image_labels` (`List[str]`): Labels for each image (e.g., subject IDs)
  - `data_type_label` (`str`): Label for the colorbar (e.g., "Contrast Estimate")

- **n_std** (`float`, optional): Threshold for outlier detection in SD units (default: `2`)

- **output_dir** (`str`, optional): Directory for saving outputs (PDF and temporary files)

### Outputs

- A [PDF](survey_medley_output_example/outlier_analysis.pdf) containing:
  1. Image slices per subject, grouped by contrast
  2. Histogram of all outlier percentages
  3. Separate histograms for each contrast
- A [CSV](survey_medley_output_example/percent_outlier_data.csv) summarizing outlier percentages per subject/contrast

🔹 Temporary files are created during processing and removed afterward.
import gc
import os
import time
from typing import List, Optional, Tuple

import img2pdf  # Make sure to install with: pip install img2pdf
import matplotlib.pyplot as plt
import numpy as np  # Added missing import
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from nilearn import datasets, plotting
from nilearn.image import load_img


def get_symmetric_percentile_bounds(
    nifti_paths: List[str], percentile: float = 98
) -> float:
    """
    Calculate symmetric percentile bounds for a set of NIfTI images.

    Args:
        nifti_paths (List[str]): List of paths to NIfTI image files.
        percentile (float): Percentile to use for calculating bounds (default: 98).

    Returns:
        float: The symmetric upper bound.
    """
    all_data = [load_img(p).get_fdata() for p in nifti_paths]
    all_data_flat = np.concatenate([d.ravel() for d in all_data])
    all_data_flat = all_data_flat[np.isfinite(all_data_flat)]  # exclude NaNs/Infs

    high = np.percentile(np.abs(all_data_flat), percentile)
    return high


def get_outlier_voxel_percentages(
    nifti_paths: List[str], n_std: float = 2
) -> List[float]:
    """
    Calculate the percentage of outlier voxels for each subject.

    Args:
        nifti_paths (List[str]): List of paths to NIfTI image files.
        n_std (float): Number of standard deviations to use for outlier detection (default: 2).

    Returns:
        List[float]: List of outlier percentages for each subject.
    """
    # Load all images into a 4D array: (n_subjects, x, y, z)
    data = np.array([load_img(p).get_fdata() for p in nifti_paths])

    # Compute voxelwise mean and std across subjects: shape (x, y, z)
    voxelwise_mean = np.mean(data, axis=0)
    voxelwise_std = np.std(data, axis=0)

    # Define voxelwise bounds
    lower_bound = voxelwise_mean - n_std * voxelwise_std
    upper_bound = voxelwise_mean + n_std * voxelwise_std

    # Define a mask
    epsilon = 1e-6
    valid_mask = (np.isfinite(voxelwise_std)) & (voxelwise_std > epsilon)

    # Compute percentage of voxels outside the bounds for each subject
    outlier_percentages = []
    for subject_data in data:
        mask = np.isfinite(subject_data) & valid_mask
        outliers = (subject_data < lower_bound) | (subject_data > upper_bound)
        valid_voxels = np.sum(mask)
        outlier_voxels = np.sum(outliers & mask)
        outlier_percentages.append(100 * outlier_voxels / valid_voxels)

    return outlier_percentages


def summarize_outlier_percentages(
    df_list: List[pd.DataFrame],
    output_dir: str,
    temp_dir: str = None,
) -> Tuple[str, str]:
    """
    Combines a list of DataFrames with subject outlier data,
    generates summary plots, and saves combined data to CSV.

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames with at least
            'subject_outlier_percentage' and 'contrast_name' columns.
        output_dir (str): Directory to save plots and CSV.

    Returns:
        Tuple[str, str]: Paths to the histogram image files.
            (1) Combined histogram
            (2) Faceted histogram by contrast
    """
    if temp_dir is None:
        temp_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    combined_df = pd.concat(df_list, ignore_index=True)

    # Plot 1: Combined histogram
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(
        combined_df['subject_outlier_percentage'],
        bins=30,
        kde=False,
        color='skyblue',
        edgecolor='black',
        ax=ax1,
    )
    ax1.set_title('Distribution of Outlier Percentages (All Contrasts)')
    ax1.set_xlabel('Subject Outlier Percentage')
    ax1.set_ylabel('Frequency')

    path_all = os.path.join(output_dir, 'outlier_percentage_dist_all.png')
    fig1.tight_layout()
    fig1.savefig(path_all, dpi=300)
    plt.close(fig1)

    # Plot 2: Faceted histogram by contrast
    g = sns.displot(
        combined_df,
        x='subject_outlier_percentage',
        col='contrast_name',
        col_wrap=5,
        bins=20,
        facet_kws={'sharex': False, 'sharey': False},
        height=3,
        aspect=1.2,
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('Subject Outlier Percentage', 'Frequency')

    path_by_contrast = os.path.join(temp_dir, 'outlier_percentage_dist_by_contrast.png')
    plt.tight_layout()
    g.savefig(path_by_contrast, dpi=300)
    plt.close(g.fig)

    # Save combined data
    csv_path = os.path.join(output_dir, 'percent_outlier_data.csv')
    combined_df.to_csv(csv_path, index=False)

    return [path_all, path_by_contrast]


def generate_png_files_sd_from_mean(
    subject_labels: List[str],
    contrast_paths: List[str],
    output_dir: str,
    contrast_name: str,
    colorbar_title=None,
    extra_labels: List[str] = None,
    n_std: float = 2,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate PNG files showing standard deviation from mean for contrast images.

    Args:
        subject_labels (List[str]): Labels for each contrast image (typically the subject ID).
        contrast_paths (List[str]): Paths to contrast image files.
        output_dir (str): Directory to save output PNG files.
        contrast_name (str): Name of the contrast for the title.
        extra_labels (List[str], optional): Additional labels for each image.
        n_std (float): Number of standard deviations to use for outlier detection (default: 2).

    Returns:
        Tuple[str, pd.DataFrame]: Path to the PNG file and a summary dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)

    subject_outlier_percentages = get_outlier_voxel_percentages(
        contrast_paths, n_std=n_std
    )
    vmax = get_symmetric_percentile_bounds(contrast_paths)
    vmin = -vmax
    mni_mask = datasets.load_mni152_brain_mask()

    fig = _plot_subject_grid(
        subject_labels,
        contrast_paths,
        subject_outlier_percentages,
        mni_mask,
        contrast_name,
        vmax,
        vmin,
        extra_labels,
        colorbar_title,
        n_std,
    )

    png_path = os.path.join(output_dir, f'{contrast_name}_slice_grid.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'Saved {png_path}')
    plt.close(fig)
    del fig
    gc.collect()
    time.sleep(0.5)

    summary_df = _build_outlier_summary_df(
        subject_labels, subject_outlier_percentages, contrast_name, extra_labels
    )

    return png_path, summary_df


def _plot_subject_grid(
    subject_labels,
    contrast_paths,
    outlier_percentages,
    mni_mask,
    contrast_name,
    vmax,
    vmin,
    extra_labels,
    colorbar_title,
    n_std,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    n_subjects = len(subject_labels)

    if n_subjects <= 12:
        ncols = 4
    elif n_subjects <= 30:
        ncols = 5
    elif n_subjects <= 60:
        ncols = 6
    else:
        ncols = 10

    nrows = int(np.ceil(n_subjects / ncols))
    subplot_width = 2.0
    subplot_height = 1.6
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height + 1.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.25)

    if n_subjects <= 20:
        title_fontsize = 9
    elif n_subjects <= 50:
        title_fontsize = 7
    else:
        title_fontsize = 5

    for i, (label, path, outlier) in enumerate(
        zip(subject_labels, contrast_paths, outlier_percentages)
    ):
        print(f'Processing contrast {i + 1}/{n_subjects}: {label}')
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        display = plotting.plot_stat_map(
            path,
            display_mode='z',
            cut_coords=[5],
            colorbar=False,
            vmax=vmax,
            vmin=vmin,
            title=None,
            axes=ax,
            bg_img=None,
            annotate=False,
        )
        display.add_contours(mni_mask, colors='greenyellow', linewidths=1.5)

        if extra_labels:
            extra_label = extra_labels[i]
            title = f'{label} {extra_label}\n({outlier:.1f}% > {n_std}SD)'
        else:
            title = f'{label}\n({outlier:.1f}% > {n_std}SD)'

        ax.set_title(title, fontsize=title_fontsize, pad=4)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cmap = get_cmap('cold_hot')
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label(colorbar_title, fontsize=10)

    fig.suptitle(f'{contrast_name}', fontsize=14, y=0.98)

    return fig


def _build_outlier_summary_df(
    subject_labels: List[str],
    subject_outlier_percentages: List[float],
    contrast_name: str,
    extra_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    data = {
        'subject_label': subject_labels,
        'subject_outlier_percentage': subject_outlier_percentages,
        'contrast_name': [contrast_name] * len(subject_labels),
        'extra_label': extra_labels if extra_labels else [None] * len(subject_labels),
    }
    return pd.DataFrame(data)


def generate_png_files_sd_from_mean_old_works(
    subject_labels: List[str],
    contrast_paths: List[str],
    output_dir: str,
    contrast_name: str,
    colorbar_title=None,
    extra_labels: List[str] = None,
    n_std: float = 2,
) -> str:
    """
    Generate PNG files showing standard deviation from mean for contrast images.

    Args:
        subject_labels (List[str]): Labels for each contrast image (typically the subject ID).
        contrast_paths (List[str]): Paths to contrast image files.
        output_dir (str): Directory to save output PNG files.
        contrast_name (str): Name of the contrast for the title.
        extra_labels (List[str], optional): Additional labels for each image.
        n_std (float): Number of standard deviations to use for outlier detection (default: 2).

    Returns:
        str: Path to the generated PNG file.
    """
    subject_outlier_percentages = get_outlier_voxel_percentages(
        contrast_paths, n_std=n_std
    )
    vmax = get_symmetric_percentile_bounds(contrast_paths)
    vmin = -vmax

    # Used to add edge contours to the images
    mni_mask = datasets.load_mni152_brain_mask()

    os.makedirs(output_dir, exist_ok=True)

    n_subjects = len(subject_labels)

    # Dynamically choose number of columns
    if n_subjects <= 12:
        ncols = 4
    elif n_subjects <= 30:
        ncols = 5
    elif n_subjects <= 60:
        ncols = 6
    else:
        ncols = 10

    nrows = int(np.ceil(n_subjects / ncols))

    # Fixed subplot size, scalable total figure size
    subplot_width = 2.0
    subplot_height = 1.6
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height + 1.5  # Extra space for suptitle and colorbar
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.25)

    # Title font size scaling
    if n_subjects <= 20:
        title_fontsize = 9
    elif n_subjects <= 50:
        title_fontsize = 7
    else:
        title_fontsize = 5

    for i, (con_label, contrast_path, subject_outlier_percentage) in enumerate(
        zip(subject_labels, contrast_paths, subject_outlier_percentages)
    ):
        print(f'Processing contrast {i + 1}/{n_subjects}: {con_label}')
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        display = plotting.plot_stat_map(
            contrast_path,
            display_mode='z',
            cut_coords=[5],
            colorbar=False,
            vmax=vmax,
            title=None,
            axes=ax,
            bg_img=None,
            annotate=False,
        )
        display.add_contours(mni_mask, colors='greenyellow', linewidths=1.5)

        if extra_labels:
            extra_label = extra_labels[i]
            ax.set_title(
                f'{con_label} {extra_label}\n({subject_outlier_percentage:.1f}% > {n_std}SD)',
                fontsize=title_fontsize,
                pad=4,
            )
        else:
            ax.set_title(
                f'{con_label}\n({subject_outlier_percentage:.1f}% > {n_std}SD)',
                fontsize=title_fontsize,
                pad=4,
            )

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cmap = get_cmap('cold_hot')
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label(colorbar_title, fontsize=10)

    fig.suptitle(f'{contrast_name}', fontsize=14, y=0.98)

    png_path = os.path.join(output_dir, f'{contrast_name}_slice_grid.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'Saved {png_path}')
    plt.close(fig)
    del fig
    gc.collect()
    time.sleep(0.5)

    return png_path


def get_mean_std_bounds(nifti_paths, n_std=2):
    all_data = [load_img(p).get_fdata() for p in nifti_paths]
    all_data_flat = np.concatenate([d.ravel() for d in all_data])
    all_data_flat = all_data_flat[np.isfinite(all_data_flat)]  # exclude NaNs/Infs

    mean = np.mean(all_data_flat)
    std = np.std(all_data_flat)

    vmin = mean - n_std * std
    vmax = mean + n_std * std
    return vmin, vmax


def combine_pngs_to_pdf(png_files, pdf_path):
    """
    Combine PNG files into a single PDF.

    Args:
        png_files: List of PNG file paths
        pdf_path: Output PDF file path
    """
    if not png_files:
        print('No PNG files to combine')
        return

    print(f'Combining {len(png_files)} PNG files into PDF: {pdf_path}')

    # Create PDF directory if it doesn't exist
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Convert to bytes for img2pdf
    with open(pdf_path, 'wb') as f:
        f.write(img2pdf.convert(png_files))

    print(f'PDF created successfully: {pdf_path}')

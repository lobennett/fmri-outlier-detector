import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.masking import apply_mask, compute_brain_mask


def get_subids_from_directory(directory: str) -> list[str]:
    """
    Given a directory, returns a list of subdirectories whose names begin with 'sub-'.

    Parameters:
    - directory (str): Path to the parent directory to search.

    Returns:
    - list[str]: A list of subdirectory names that start with 'sub-'.
    """
    return [
        name
        for name in os.listdir(directory)
        if name.startswith('sub-') and os.path.isdir(os.path.join(directory, name))
    ]


def get_task_names_from_directory(data_root: str) -> list[str]:
    """
    Given a directory, returns a sorted list of unique subdirectory names
    (assumed to be task names) found inside any 'sub-*' directory.
    """
    task_dirs = glob(os.path.join(data_root, 'sub-*', '*'))
    task_names = {os.path.basename(path) for path in task_dirs if os.path.isdir(path)}
    return sorted(task_names)


def extract_session_from_filename(filename: str) -> str:
    """Extract session from filename.

    Args:
        filename: Filename containing session information

    Returns:
        str: Extracted session ID or 'unknown' if not found
    """
    return (
        filename.split('_ses-')[1].split('_')[0] if '_ses-' in filename else 'unknown'
    )


def load_vif_data(vif_file: Path, target_contrast: str) -> float:
    """Load VIF data from file and extract VIF value for target contrast.

    Args:
        vif_file: Path to VIF file
        target_contrast: Target contrast name

    Returns:
        float: VIF value for target contrast
    """
    vif_data = pd.read_csv(vif_file)
    return vif_data[vif_data['contrast'] == target_contrast]['VIF'].values[0]


def get_task_baseline_contrasts(task_name: str) -> str:
    """Get task-baseline contrast provided the task name"""
    contrasts = {
        'cuedTS': (
            '1/3*(task_stay_cue_switch+task_stay_cue_stay+task_switch_cue_switch)'
        ),
        'spatialTS': (
            '1/3*(task_stay_cue_switch+task_stay_cue_stay+task_switch_cue_switch)'
        ),
        'directedForgetting': '1/4*(con+pos+neg+memory_and_cue)',
        'flanker': '1/2*congruent + 1/2*incongruent',
        'goNogo': '1/2*go+1/2*nogo_success',
        'nBack': '1/4*(mismatch_1back+match_1back+mismatch_2back+match_2back)',
        'stopSignal': '1/3*go + 1/3*stop_failure + 1/3*stop_success',
        'shapeMatching': '1/7*(SSS+SDD+SNN+DSD+DDD+DDS+DNN)',
        'directedForgettingWFlanker': (
            '1/7*(congruent_pos+congruent_neg+congruent_con+incongruent_pos+'
            'incongruent_neg+incongruent_con+memory_and_cue)'
        ),
        'stopSignalWDirectedForgetting': (
            '1/10*(go_pos+go_neg+go_con+stop_success_pos+stop_success_neg+'
            'stop_success_con+stop_failure_pos+stop_failure_neg+stop_failure_con+'
            'memory_and_cue)'
        ),
        'stopSignalWFlanker': (
            '1/6*(go_congruent+go_incongruent+stop_success_congruent+'
            'stop_success_incongruent+stop_failure_congruent+stop_failure_incongruent)'
        ),
    }
    return contrasts[task_name]


def get_target_contrast(contrast: str, task_name: str) -> str:
    """Get short name of contrast provided the full contrast name"""
    contrasts = {
        'task-baseline': get_task_baseline_contrasts(task_name),
        'main_vars': '1/3*(SDD+DDD+DDS)-1/2*(SNN+DNN)',
        'cue_switch_cost': 'task_stay_cue_switch-task_stay_cue_stay',
        'task_switch_cost': 'task_switch_cue_switch-task_stay_cue_switch',
    }
    return contrasts.get(contrast, contrast)


def get_unique_contrasts(
    indiv_contrasts_dir: Path, subj_id: str, task_name: str
) -> list[str]:
    """Get unique contrasts from effect size files"""
    all_contrasts = glob(f'{indiv_contrasts_dir}/*{subj_id}*{task_name}*effect-size*')
    contrasts = [
        contrast.split('_contrast-')[1].split('_rtmodel')[0]
        for contrast in all_contrasts
    ]
    return np.unique(contrasts)


def fetch_contrast_zstat_vif(
    contrast: str,
    indiv_contrasts_dir: Path,
    quality_control_dir: Path,
    subj_id: str,
    task_name: str,
) -> Dict[str, List]:
    """Retreive VIF and zstat data for a contrast.

    Args:
        contrast: Contrast name
        indiv_contrasts_dir: Directory containing contrast files
        quality_control_dir: Directory containing quality control files
        subj_id: Subject ID
        task_name: Name of the task

    Returns:
        dict: Dictionary containing contrast specific data including subid, session,
        vif, contrast name and zstat paths
    """
    # Get zstat files
    contrast_zstat_files = glob(
        f'{indiv_contrasts_dir}/*{subj_id}*{task_name}*contrast-{contrast}_rtmodel*z_score*'
    )

    # Extract session numbers from filenames
    sessions = []
    vifs = []

    for eff_size in contrast_zstat_files:
        # Extract session from filename
        session_match = extract_session_from_filename(eff_size)
        sessions.append(session_match)

        # Load VIF data
        vif_file = list(
            quality_control_dir.glob(f'*{subj_id}*{session_match}*{task_name}*')
        )
        assert len(vif_file) == 1, f'Expected 1 VIF file, found {len(vif_file)}'

        # Get VIF value
        target_contrast = get_target_contrast(contrast, task_name)
        vif_value = load_vif_data(vif_file[0], target_contrast)
        vifs.append(vif_value)

    return {
        'zstat_files': contrast_zstat_files,
        'contrast_names': [contrast] * len(contrast_zstat_files),
        'sessions': sessions,
        'vifs': vifs,
    }


def fetch_all_contrasts_zstats_vifs(
    task_name: str,
    subj_id: str,
    data_root: str,
) -> Dict[str, Any]:
    """Get zstats and VIFs across all sessions/subjects/contrasts for a specific task

    Args:
        task_name: Name of the task
        subj_id: Subject ID
        data_root: Root directory for data

    Returns:
        dict: Dictionary containing zstat paths, zstats, contrast names, vifs,
              subject and session data
    """
    # Paths
    # - Input path
    subj_lev1_dir = Path(f'{data_root}/{subj_id}/{task_name}')
    # - Contains effect size files for subject
    indiv_contrasts_dir = Path(f'{subj_lev1_dir}/indiv_contrasts')
    # - Contains files with corresponding VIF values for subject
    quality_control_dir = Path(f'{subj_lev1_dir}/quality_control')

    # Get unique contrasts
    unique_contrasts = get_unique_contrasts(indiv_contrasts_dir, subj_id, task_name)

    # carefully concatenate contrasts and variance images to keep order consistent
    all_zstat_files, all_con_names, all_sessions, all_vifs = (
        [],
        [],
        [],
        [],
    )
    # Process each contrast
    for contrast in unique_contrasts:
        contrast_data = fetch_contrast_zstat_vif(
            contrast, indiv_contrasts_dir, quality_control_dir, subj_id, task_name
        )

        all_zstat_files.extend(contrast_data['zstat_files'])
        all_con_names.extend(contrast_data['contrast_names'])
        all_sessions.extend(contrast_data['sessions'])
        all_vifs.extend(contrast_data['vifs'])

    return {
        'zstat_files': all_zstat_files,
        'contrast_names': all_con_names,
        'session_names': all_sessions,
        'vifs': all_vifs,
    }


def all_tasks_fetch_all_contrasts_zstats_vifs(
    data_root: str,
) -> pd.DataFrame:
    """
    For all tasks, all subjects and contrast names are fetched. For each contrast
    within each task, the zstat files and VIFs are fetched for all subjects/sessions.
    """
    subj_ids = get_subids_from_directory(data_root)
    task_names = get_task_names_from_directory(data_root)

    records = []

    for task_name in task_names:
        for subj_id in subj_ids:
            try:
                output = fetch_all_contrasts_zstats_vifs(task_name, subj_id, data_root)
            except Exception as e:
                print(f'Skipping {task_name}, {subj_id} due to error: {e}')
                continue

            # Round the VIFs
            vifs_rounded = [np.round(val, 2) for val in output['vifs']]

            # Create records for each contrast
            for contrast_name, vif, zstat_file, session in zip(
                output['contrast_names'],
                vifs_rounded,  # Use the rounded VIFs list
                output['zstat_files'],
                output['session_names'],
            ):
                records.append(
                    {
                        'subid': subj_id,
                        'task': task_name,
                        'contrast_names': contrast_name,
                        'vifs': vif,
                        'zstat_files': zstat_file,
                        'session': session,
                    }
                )
    df = pd.DataFrame(records)

    # Add additional columns
    df['task_contrast'] = df.apply(
        lambda row: f'{row["task"]}: {row["contrast_names"]}', axis=1
    )
    df['subid_ses_vif'] = df.apply(
        lambda row: f'{row["subid"]}_ses_{row["session"]}: vif={row["vifs"]}', axis=1
    )

    return df


def concatenate_mask_images(img_files: List[Union[str, nib.Nifti1Image]]) -> np.ndarray:
    """
    Concatenates a list of 3D NIfTI images into a single 4D image,
    computes a brain mask, and applies the mask to extract voxel values.

    Parameters
    ----------
    img_files : list of str or nib.Nifti1Image
        A list of NIfTI image file paths or Nifti1Image objects.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_images, n_voxels) with voxel data for each image.
    """
    concatenated_img = nib.concat_images(img_files)
    brain_mask = compute_brain_mask(concatenated_img)
    masked_data = apply_mask(concatenated_img, brain_mask)
    return np.asarray(masked_data)


def plot_intensity_grid(data, data_row_labels, plot_title):
    fig, ax = plt.subplots(figsize=(6, 7.5))
    sns.heatmap(
        data,
        cmap='RdBu_r',
        center=0,
        yticklabels=data_row_labels,
        cbar_kws={'label': 'Signal Intensity'},
        ax=ax,
    )
    ax.set_title(f'Contrast: {plot_title}')
    ax.set_xlabel('Voxels')
    ax.set_ylabel('Subjects (VIFs)')
    ax.set_xticks([])
    return fig, ax


def compute_extreme_voxel_percentages(zstat_data: np.ndarray, n_sd: int) -> np.ndarray:
    means = np.mean(zstat_data, axis=0)
    sds = np.std(zstat_data, axis=0)
    lower = means - n_sd * sds
    upper = means + n_sd * sds
    is_extreme = (zstat_data < lower) | (zstat_data > upper)
    return 100 * np.sum(is_extreme, axis=1) / zstat_data.shape[1]


def summarize_extreme_voxels(zstat_vif_data: pd.DataFrame) -> pd.DataFrame:
    extreme_voxels_summary = []
    contrast_labels = zstat_vif_data['task_contrast'].unique()

    for i, con_label in enumerate(contrast_labels):
        print(f'Processing {i + 1}/{len(contrast_labels)}: {con_label}')

        filtered_df = zstat_vif_data[zstat_vif_data['task_contrast'] == con_label]
        zstat_masked_2d = concatenate_mask_images(filtered_df['zstat_files'])

        percent_extreme_2sd = compute_extreme_voxel_percentages(zstat_masked_2d, 2)
        percent_extreme_3sd = compute_extreme_voxel_percentages(zstat_masked_2d, 3)

        result_df = pd.DataFrame(
            {
                'contrast_label': filtered_df['task_contrast'].values,
                'subid_vif': filtered_df['subid_ses_vif'].values,
                'percent_extreme_voxels_2sd': percent_extreme_2sd,
                'percent_extreme_voxels_3sd': percent_extreme_3sd,
            }
        )

        extreme_voxels_summary.append(result_df)

    return pd.concat(extreme_voxels_summary, ignore_index=True)

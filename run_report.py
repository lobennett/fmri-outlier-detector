import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import image
from typing import Union, Optional, Set, Tuple
import re
from collections import defaultdict

def get_flag_reason(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """
    Checks a DataFrame against flagging criteria
    """
    conditions = [
        (df['VIF'] >= thresholds['vif_strict'])
        & (df['image_outlier_percentage'] >= thresholds['outlier_strict']),
        df['VIF'] >= thresholds['vif_strict'],
        df['image_outlier_percentage'] >= thresholds['outlier_strict'],
        (df['VIF'] >= thresholds['vif_combined'])
        & (df['image_outlier_percentage'] >= thresholds['outlier_combined']),
    ]
    reasons = [
        f'VIF >= {thresholds["vif_strict"]}; Outlier Pct >= {thresholds["outlier_strict"]}%',
        f'VIF >= {thresholds["vif_strict"]}',
        f'Outlier Pct >= {thresholds["outlier_strict"]}%',
        f'VIF >= {thresholds["vif_combined"]} & Outlier >= {thresholds["outlier_combined"]}%',
    ]
    return np.select(conditions, reasons, default='')


def summarize_flagged_scans(flagged_df: pd.DataFrame):
    """
    Groups flagged scans by contrast and returns a summary DataFrame.
    """
    if flagged_df.empty:
        print('No scans were flagged. No summary to generate.')
        return pd.DataFrame()

    contrast_counts = (
        flagged_df.groupby('contrast_name')
        .size()
        .reset_index(name='flagged_scan_count')
    )
    contrast_counts = contrast_counts.sort_values(
        by='flagged_scan_count', ascending=False
    )

    total_flags = contrast_counts['flagged_scan_count'].sum()
    total_row = pd.DataFrame(
        [{'contrast_name': 'Total', 'flagged_scan_count': total_flags}]
    )

    summary_df = pd.concat([contrast_counts, total_row], ignore_index=True)
    return summary_df


def summarize_flagged_by_category(flagged_df: pd.DataFrame, thresholds: dict):
    """
    Summarize flagged scans by category: VIF only, Outliers only, or Both.
    """
    if flagged_df.empty:
        print('No scans were flagged for category breakdown.')
        return pd.DataFrame()

    # Define categories based on thresholds
    vif_only = (flagged_df['VIF'] >= thresholds['vif_strict']) & (flagged_df['image_outlier_percentage'] < thresholds['outlier_strict'])
    outliers_only = (flagged_df['VIF'] < thresholds['vif_strict']) & (flagged_df['image_outlier_percentage'] >= thresholds['outlier_strict'])
    both = (flagged_df['VIF'] >= thresholds['vif_strict']) & (flagged_df['image_outlier_percentage'] >= thresholds['outlier_strict'])

    category_counts = pd.DataFrame([
        {'category': f'VIF >= {thresholds["vif_strict"]} only', 'count': vif_only.sum()},
        {'category': f'Outliers >= {thresholds["outlier_strict"]}% only', 'count': outliers_only.sum()},
        {'category': f'Both (VIF >= {thresholds["vif_strict"]} & Outliers >= {thresholds["outlier_strict"]}%)', 'count': both.sum()},
        {'category': 'Total', 'count': len(flagged_df)}
    ])
    
    return category_counts


def create_parser():
    """
    Parser to customize input/output directories and flag thresholds
    """
    parser = argparse.ArgumentParser(
        description='Flag and summarize outlier scans from imaging data.'
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help="Path to the 'percent_outlier_data.csv' file.",
    )
    parser.add_argument(
        '--level1-dir',
        type=Path,
        help="Path to first level maps (required only if not using --skip-r-value).",
    ) 
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory to save the output 'flagged.csv' and 'summary.csv' files.",
    )
    parser.add_argument(
        '--vif-threshold-strict',
        type=int,
        default=15,
        help='VIF threshold for strict flagging.',
    )
    parser.add_argument(
        '--outlier-threshold-strict',
        type=int,
        default=15,
        help='Outlier percentage threshold for strict flagging.',
    )
    parser.add_argument(
        '--vif-threshold-combined',
        type=int,
        default=10,
        help='VIF threshold for combined flagging.',
    )
    parser.add_argument(
        '--outlier-threshold-combined',
        type=int,
        default=10,
        help='Outlier percentage threshold for combined flagging.',
    )
    parser.add_argument(
        '--skip-r-value',
        action='store_true',
        help='Skip within-subject r-value similarity calculation.',
    )
    return parser

def create_summary(args, thresholds):
    try:
        # Load and preprocess data
        input_file = args.input_file
        if not input_file.exists():
            raise FileNotFoundError(f'Input file not found at: {input_file}')

        df = pd.read_csv(input_file)
        print('Successfully loaded and preprocessed data.')

        crit1 = (df['VIF'] >= thresholds['vif_combined']) & (
            df['image_outlier_percentage'] >= thresholds['outlier_combined']
        )
        crit2 = df['VIF'] >= thresholds['vif_strict']
        crit3 = df['image_outlier_percentage'] >= thresholds['outlier_strict']
        combined_mask = crit1 | crit2 | crit3
        flagged_df = df[combined_mask].copy()

        print(f'\n--- Flagged Scans ({len(flagged_df)} total) ---')
        print(
            f'Criteria: (VIF >= {thresholds["vif_strict"]}) OR '
            f'(Outlier >= {thresholds["outlier_strict"]}%) OR '
            f'(VIF >= {thresholds["vif_combined"]} & Outlier >= {thresholds["outlier_combined"]}%)'
        )

        if not flagged_df.empty:
            # Get reasons and sort the final output
            flagged_df['flag_reason'] = get_flag_reason(flagged_df, thresholds)

            flagged_df['subject_num'] = (
                flagged_df['subject_label'].str.extract(r'sub-s(\d+)').astype(int)
            )
            flagged_df['session_num'] = flagged_df['session_id'].astype(int)

            sorted_df = flagged_df.sort_values(
                by=['subject_num', 'task_name', 'session_num']
            )

            final_columns = [
                'subject_label',
                'task_name',
                'contrast_name',
                'image_outlier_percentage',
                'VIF',
                'flag_reason',
            ]
            final_df = sorted_df[final_columns]

            # Save flagged scans
            args.output_dir.mkdir(parents=True, exist_ok=True)
            flagged_fpath = args.output_dir / 'flagged_scans.csv'
            final_df.to_csv(flagged_fpath, index=False)
            print(f'\nSaved {len(final_df)} flagged scans to: {flagged_fpath}')

            # Summarize and save summary
            summary_df = summarize_flagged_scans(flagged_df)
            if not summary_df.empty:
                summary_fpath = args.output_dir / 'flagged_summary.csv'
                summary_df.to_csv(summary_fpath, index=False)
                print(f'Saved summary to: {summary_fpath}')
                print('\n--- Summary of Flagged Scans by Contrast ---')
                print(summary_df.to_string(index=False))
            
            # Category breakdown summary
            category_df = summarize_flagged_by_category(flagged_df, thresholds)
            if not category_df.empty:
                category_fpath = args.output_dir / 'flagged_category_summary.csv'
                category_df.to_csv(category_fpath, index=False)
                print(f'Saved category breakdown to: {category_fpath}')
                print('\n--- Summary of Flagged Scans by Category ---')
                print(category_df.to_string(index=False))
        else:
            print('No scans met the flagging criteria.')

    except FileNotFoundError as e:
        print(f'Error: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def extract_subject_id(filename: str) -> Optional[str]:
    """
    Extract subject ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Subject ID or None if not found
    """
    match = re.search(r'(sub-s\d+)', filename)
    return match.group(1) if match else None


def extract_session_id(filename: str) -> Optional[str]:
    """
    Extract session ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Session ID or None if not found
    """
    match = re.search(r'(ses-\d+)', filename)
    return match.group(1) if match else None


def extract_contrast_name(filename: str) -> Optional[str]:
    """
    Extract contrast name from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Contrast name or None if not found
    """
    # Extract task name
    task_match = re.search(r'task-([^_]+)', filename)
    if not task_match:
        return None
    task = task_match.group(1)
    
    # Extract full contrast name (everything between 'contrast-' and '_rtmodel-' or '_stat-')
    contrast_start = filename.find('contrast-')
    if contrast_start == -1:
        return None
    
    contrast_start += len('contrast-')
    
    # Find the end of the contrast (before '_rtmodel-rt_' or '_stat-')
    rtmodel_pos = filename.find('_rtmodel-', contrast_start)
    stat_pos = filename.find('_stat-', contrast_start)
    
    # Use the earliest valid position
    end_positions = [pos for pos in [rtmodel_pos, stat_pos] if pos > contrast_start]
    if not end_positions:
        return None
    
    contrast_end = min(end_positions)
    contrast = filename[contrast_start:contrast_end]
    
    return f'task-{task}_contrast-{contrast}'


def extract_run_id(filename: str) -> Optional[str]:
    """
    Extract run ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Run ID or None if not found
    """
    match = re.search(r'run-(\d+)', filename)
    if match:
        run_num = match.group(1)
        # Ensure it's zero-padded to 2 digits
        return f'run-{run_num.zfill(2)}'
    return None


def extract_contrast_info(
    filepath: Path,
) -> dict:
    """
    Extract subject, session, run, and task contrast information from filepath.
    
    Parameters
    ----------
    filepath : Path
        Path to the contrast file
        
    Returns
    -------
    dict
        Dictionary with keys: sub, ses_str, ses_num, contrast, run_str, run_num
    """
    filename = filepath.name
    
    sub = extract_subject_id(filename)
    ses_str = extract_session_id(filename)
    contrast = extract_contrast_name(filename)
    run_str = extract_run_id(filename)
    
    # Extract numeric values
    ses_num = None
    if ses_str:
        ses_match = re.search(r'ses-(\d+)', ses_str)
        if ses_match:
            ses_num = int(ses_match.group(1))
    
    run_num = None
    if run_str:
        run_match = re.search(r'run-(\d+)', run_str)
        if run_match:
            run_num = int(run_match.group(1))
    
    return {
        'sub': sub,
        'ses_str': ses_str,
        'ses_num': ses_num,
        'contrast': contrast,
        'run_str': run_str,
        'run_num': run_num
    }


def compute_within_subject_contrast_similarity(level1_dir: Path, output_dir: Path, input_file: Path):
    """
    Computes within-subject similarity for each contrast separately.

    For each subject and each contrast:
    1. Finds all maps for that specific contrast
    2. Creates a mask from the mean of those contrast maps
    3. Computes mean r-value for each map compared to all other maps of same contrast
    4. Merges with VIF and outlier data from input file
    5. Saves results sorted by numeric subject ID, session, then task name

    Args:
        level1_dir (Path): Directory containing first level maps
        output_dir (Path): Directory to save output CSV
        input_file (Path): Path to percent_outlier_data.csv file
    """
    print("Computing within-subject contrast similarity...")
    
    # Load VIF and outlier data
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    vif_data = pd.read_csv(input_file)
    print(f"Loaded VIF/outlier data: {len(vif_data)} rows")
    
    # Find all contrast files
    contrast_files = list(level1_dir.glob("**/indiv_contrasts/*stat-effect-size.nii.gz"))
    if not contrast_files:
        print(f"No contrast files found in {level1_dir}")
        return

    print(f"Found {len(contrast_files)} contrast files")
    
    # Group files by subject and contrast
    subject_contrast_groups = defaultdict(lambda: defaultdict(list))
    
    for file_path in contrast_files:
        info = extract_contrast_info(file_path)
        if info['sub'] and info['contrast']:
            subject_contrast_groups[info['sub']][info['contrast']].append({
                'path': file_path,
                'info': info
            })
    
    all_results = []
    
    for subject, contrasts in subject_contrast_groups.items():
        print(f"Processing subject {subject}...")
        
        for contrast_name, contrast_files_info in contrasts.items():
            if len(contrast_files_info) < 2:
                print(f"  Skipping {contrast_name}: only {len(contrast_files_info)} file(s)")
                continue
                
            print(f"  Processing {contrast_name} ({len(contrast_files_info)} maps)")
            
            # Load all images for this contrast
            image_objects = [image.load_img(f['path']) for f in contrast_files_info]
            
            # Create mask from mean of contrast maps
            mean_img = image.mean_img(image_objects, copy_header=True)
            mask_data = mean_img.get_fdata() != 0
            
            if np.sum(mask_data) == 0:
                print(f"    Warning: Empty mask for {contrast_name}")
                continue
            
            # Extract data for each map
            maps_data = []
            for i, img_obj in enumerate(image_objects):
                img_data = img_obj.get_fdata()
                beta_values = img_data[mask_data]
                
                info = contrast_files_info[i]['info']
                maps_data.append({
                    'beta_values': beta_values,
                    'info': info
                })
            
            # Compute mean r-value for each map compared to all others
            for i, current_map in enumerate(maps_data):
                correlations = []
                
                for j, other_map in enumerate(maps_data):
                    if i != j:
                        r_value = np.corrcoef(current_map['beta_values'], 
                                            other_map['beta_values'])[0, 1]
                        if not np.isnan(r_value):
                            correlations.append(r_value)
                
                if correlations:
                    mean_r = np.mean(correlations)
                    
                    # Extract task name from contrast
                    task_match = re.search(r'task-([^_]+)', contrast_name)
                    task_name = task_match.group(1) if task_match else 'unknown'
                    
                    # Extract numeric IDs for sorting
                    sub_num = None
                    if current_map['info']['sub']:
                        sub_match = re.search(r'sub-s(\d+)', current_map['info']['sub'])
                        if sub_match:
                            sub_num = int(sub_match.group(1))
                    
                    # Create lookup key for VIF data matching
                    # VIF data format: sub-s03_ses-02_run-1
                    session_str = current_map['info']['ses_str']
                    run_str = current_map['info']['run_str']
        
                    if session_str and run_str:
                        # Convert ses-02 to 02 and run-01 to 1 to match VIF format
                        ses_num_str = session_str.replace('ses-', '')
                        run_num_str = run_str.replace('run-', '').lstrip('0') or '0'
                       
                        vif_lookup_key = f"{current_map['info']['sub']}_{session_str}_run-{run_num_str}"

                        # Find matching VIF data
                        vif_match = vif_data[
                            (vif_data['subject_label'] == vif_lookup_key) &
                            (vif_data['contrast_name'] == contrast_name.replace('task-', '', 1).replace('_contrast-', '_'))
                        ]
                        
                        # Get VIF and outlier values
                        vif_value = vif_match['VIF'].iloc[0] if len(vif_match) > 0 else None
                        outlier_pct = vif_match['image_outlier_percentage'].iloc[0] if len(vif_match) > 0 else None
                    else:
                        vif_value = None
                        outlier_pct = None
                    
                    all_results.append({
                        'subject_id': current_map['info']['sub'],
                        'subject_num': sub_num,
                        'session_id': current_map['info']['ses_str'],
                        'session_num': current_map['info']['ses_num'],
                        'task_name': task_name,
                        'contrast_name': contrast_name,
                        'run_id': current_map['info']['run_str'],
                        'mean_r_value': mean_r,
                        'n_comparisons': len(correlations),
                        'VIF': vif_value,
                        'image_outlier_percentage': outlier_pct
                    })
    
    if not all_results:
        print("No similarity results computed")
        return
    
    # Create DataFrame and sort
    results_df = pd.DataFrame(all_results)
    
    # Sort by numeric subject ID, task, contrast, then session
    results_df = results_df.sort_values([
        'subject_num', 
        'task_name',
        'contrast_name',
        'session_num',
    ], na_position='last')
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'within_subject_similarity.csv'
    
    # Select and order columns for output
    output_columns = [
        'subject_id', 'session_id', 'task_name', 'contrast_name', 
        'run_id', 'mean_r_value', 'n_comparisons', 'VIF', 'image_outlier_percentage'
    ]
    
    results_df[output_columns].to_csv(output_path, index=False)
    print(f"Saved within-subject similarity results to: {output_path}")
    print(f"Total results: {len(results_df)}")
    
    return results_df

def main():
    """
    Main execution pipeline to load, flag, and summarize outlier scans.
    """

    args = create_parser().parse_args()
    
    # Validate arguments
    if not args.skip_r_value and not args.level1_dir:
        print("Error: --level1-dir is required unless --skip-r-value is used")
        return

    thresholds = {
        'vif_strict': args.vif_threshold_strict,
        'outlier_strict': args.outlier_threshold_strict,
        'vif_combined': args.vif_threshold_combined,
        'outlier_combined': args.outlier_threshold_combined,
    }

    create_summary(args, thresholds)
    
    if not args.skip_r_value:
        compute_within_subject_contrast_similarity(args.level1_dir, args.output_dir, args.input_file)
    else:
        print("Skipping r-value calculation (--skip-r-value flag used)")


if __name__ == '__main__':
    main()

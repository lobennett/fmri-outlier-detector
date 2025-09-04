import argparse
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from plotting_functions import generate_all_data_summaries


def parse_bids_entities(path: str) -> Dict[str, Optional[str]]:
    """
    Extracts BIDS-like entities from a filepath using robust, non-greedy patterns.
    """
    filename = os.path.basename(path)
    entities = {
        'subject': None,
        'session': None,
        'run': None,
        'task': None,
        'contrast': None,
        'sub_ses_key': None,
    }

    sub_match = re.search(r'sub-s([^_]+)', filename)
    ses_match = re.search(r'ses-([^_]+)', filename)
    run_match = re.search(r'run-([^_]+)', filename)
    task_match = re.search(r'task-([^_]+)', filename)
    contrast_match = re.search(r'contrast-(.+?)_rtmodel', filename)

    if sub_match:
        entities['subject'] = sub_match.group(1)
    if ses_match:
        entities['session'] = ses_match.group(1)
    if run_match:
        entities['run'] = run_match.group(1)
    if task_match:
        entities['task'] = task_match.group(1)
    if contrast_match:
        entities['contrast'] = contrast_match.group(1)

    if entities['subject'] and entities['session']:
        entities['sub_ses_key'] = (
            f'sub-s{entities["subject"]}_ses-{entities["session"]}'
        )

    return entities


def group_paths_by_filename_pattern(file_paths: List[str]) -> Dict[str, List[str]]:
    """Groups file paths by filename patterns, normalizing identifiers."""
    grouped = defaultdict(list)
    for path in file_paths:
        filename = os.path.basename(path)
        pattern = re.sub(r'sub-s[^_]+', 'sub-sXXX', filename)
        pattern = re.sub(r'ses-[^_]+', 'ses-XX', pattern)
        pattern = re.sub(r'run-[^_]+', 'run-X', pattern)
        grouped[pattern].append(path)
    return dict(sorted(grouped.items()))


def extract_vif_from_csv(csv_path: str) -> Dict[str, float]:
    """Extracts VIF data from a contrast VIF CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['contrast'], df['VIF']))
    except Exception as e:
        print(f'Warning: Could not read VIF data from {csv_path}: {e}')
        return {}


def find_vif_files(base_dir: str) -> Dict[str, Dict[str, float]]:
    """Finds and reads all VIF CSV files, returning a nested dictionary."""
    vif_pattern = f'{base_dir}/sub-s*/*/quality_control/*_contrast_vifs.csv'
    vif_files = glob(vif_pattern)
    print(f'Found {len(vif_files)} VIF files.')

    all_vif_data = {}
    for vif_file in vif_files:
        entities = parse_bids_entities(vif_file)
        if entities['sub_ses_key'] and entities['task']:
            key = f'{entities["sub_ses_key"]}_{entities["task"]}'
            all_vif_data[key] = extract_vif_from_csv(vif_file)
    return all_vif_data


def get_contrast_vif_labels(
    vif_data: Dict[str, Dict[str, float]], nifti_paths: List[str], contrast_name: str
) -> List[str]:
    """Generates VIF labels for a list of NIfTI paths."""
    vif_labels = []
    for path in nifti_paths:
        filename = os.path.basename(path)
        entities = parse_bids_entities(path)
        label = '(vif=?)'

        if entities['sub_ses_key'] and entities['task']:
            vif_key = f'{entities["sub_ses_key"]}_{entities["task"]}'

            if vif_key in vif_data:
                if contrast_name in vif_data[vif_key]:
                    vif_value = vif_data[vif_key][contrast_name]
                    label = f'(vif={vif_value:.2f})'
                else:
                    print(f'\n[DEBUG] VIF MISS for {filename}:')
                    print(f"  > Looking for contrast: '{contrast_name}'")
                    print(f'  > Available contrasts: {list(vif_data[vif_key].keys())}')
            else:
                print(
                    f"\n[DEBUG] VIF KEY NOT FOUND for {filename}: > Generated Key: '{vif_key}'"
                )
        vif_labels.append(label)
    return vif_labels


def find_nifti_files(base_dir: str) -> List[str]:
    """Finds all stat-effect-size.nii.gz files."""
    nifti_pattern = f'{base_dir}/sub-s*/*/indiv_contrasts/*stat-effect-size.nii.gz'
    return sorted(glob(nifti_pattern))


def make_list_of_input_dicts(base_dir: str) -> List[Dict[str, object]]:
    """Creates a list of dictionaries, one for each contrast to be processed."""
    print(f'Starting data preparation with base_dir: {base_dir}')

    nifti_files = find_nifti_files(base_dir)
    print(f'Found {len(nifti_files)} total NIfTI files.')

    vif_data = find_vif_files(base_dir)
    print(f'Found VIF data for {len(vif_data)} subject-session-task combinations.')

    grouped_lists = group_paths_by_filename_pattern(nifti_files)
    list_of_input_dicts = []

    for pattern, file_paths in grouped_lists.items():
        if not file_paths:
            continue

        first_file_entities = parse_bids_entities(file_paths[0])
        task_name = first_file_entities.get('task')
        contrast_name = first_file_entities.get('contrast')

        if not task_name or not contrast_name:
            print(
                f'Warning: Could not extract task or contrast from files in group: {pattern}'
            )
            continue

        nifti_paths = sorted(file_paths)
        vif_labels = get_contrast_vif_labels(vif_data, nifti_paths, contrast_name)

        path_entities = [parse_bids_entities(p) for p in nifti_paths]
        sub_ids = [e.get('subject', 'N/A') for e in path_entities]
        session_ids = [e.get('session', 'N/A') for e in path_entities]
        run_ids = [e.get('run', 'N/A') for e in path_entities]

        # CORRECTED: Create two separate lists now
        image_labels = [
            f'sub-s{sub_id}_ses-{ses_id}_run-{run_id}'
            for sub_id, ses_id, run_id in zip(sub_ids, session_ids, run_ids)
        ]

        list_of_input_dicts.append(
            {
                'main_title': f'{task_name}_{contrast_name}',
                'nifti_paths': nifti_paths,
                'image_labels': image_labels,
                'vif_labels': vif_labels,  # Add the new VIF labels list
                'data_type_label': 'Contrast Estimate',
                'task_name': task_name,
                'contrast_name': contrast_name,
                'session_ids': session_ids,
            }
        )

    print(f'Created {len(list_of_input_dicts)} input dictionaries for plotting.')
    return list_of_input_dicts


def process_contrasts(base_dir: str, output_dir: str) -> None:
    """Main pipeline for processing contrasts and generating summaries."""
    dicts_list = make_list_of_input_dicts(base_dir)
    if dicts_list:
        generate_all_data_summaries(dicts_list, output_dir=output_dir)
    else:
        print('No data dictionaries were created; skipping summary generation.')


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process fMRI contrast files and generate a QA outlier analysis report.'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing the first-level analysis output.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output PDF and CSV files will be saved.',
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    process_contrasts(args.base_dir, args.output_dir)


if __name__ == '__main__':
    main()

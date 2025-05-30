import argparse
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from plotting_functions import generate_all_data_summaries


def get_subids_from_paths(list_of_paths: List[str]) -> List[str]:
    """
    Extracts subject IDs from a list of file paths.

    Args:
        list_of_paths (List[str]): A list of file paths containing 'sub-XXX' or 'sub_XXX'.

    Returns:
        List[str]: A list of extracted subject IDs as strings.
    """
    sub_ids = []
    for path in list_of_paths:
        match = re.search(r'sub[-_]?(\d+)', path)
        if match:
            sub_ids.append(match.group(1))
    return sub_ids


def group_paths_by_filename_pattern(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Groups file paths by filename patterns, normalizing subject IDs to 'sub_XXX'.

    Args:
        file_paths (List[str]): A list of file paths.

    Returns:
        Dict[str, List[str]]: A dictionary with normalized filename patterns as keys and lists of paths as values.
    """
    grouped = defaultdict(list)
    for path in file_paths:
        filename = os.path.basename(path)
        pattern = re.sub(r'sub[-_]\d+', 'sub_XXX', filename)
        grouped[pattern].append(path)
    return dict(sorted(grouped.items()))


def extract_vif_table(html_path: str) -> pd.DataFrame:
    """
    Extracts VIF data from an HTML file into a DataFrame indexed by subject ID.

    Args:
        html_path (str): Path to the HTML file.

    Returns:
        pd.DataFrame: A DataFrame of VIF values with subjects as the index.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    all_rows = []

    th_elements = soup.find_all('th', string='contrast')
    for th in th_elements:
        tr = th.find_parent('tr')
        if not tr:
            continue
        ths = tr.find_all('th')
        if (
            len(ths) >= 2
            and ths[0].text.strip() == 'contrast'
            and ths[1].text.strip() == 'VIF'
        ):
            table = tr.find_parent('table')
            if not table:
                continue

            subject_id = None
            for prev in table.previous_elements:
                if getattr(prev, 'name', None) == 'h2' and prev.text.startswith(
                    'Subject'
                ):
                    subject_id = prev.text.split(' ')[1]
                    break

            if subject_id is None:
                continue

            vif_data = {}
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    contrast = cols[0].text.strip()
                    vif = float(cols[1].text.strip())
                    vif_data[contrast] = vif

            vif_data['subject'] = subject_id
            all_rows.append(vif_data)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.set_index('subject', inplace=True)
    return df


def get_contrast_vif_labels(
    vif_df: pd.DataFrame, sub_labels: List[str], contrast_name: str
) -> List[str]:
    """
    Generates VIF labels for given subject IDs and contrast name.

    Args:
        vif_df (pd.DataFrame): DataFrame with VIF values.
        sub_labels (List[str]): List of subject IDs.
        contrast_name (str): Contrast name to lookup VIF values.

    Returns:
        List[str]: List of formatted VIF label strings.
    """
    vif_label = []
    for sub in sub_labels:
        vif_loop = np.round(vif_df.loc[sub, contrast_name]).astype(int)
        vif_label.append(f'(vif={vif_loop})')
    return vif_label


def make_list_of_input_dicts(
    contrast_dir: str, html_path: str
) -> List[Dict[str, object]]:
    """
    Creates a list of dictionaries representing input data for processing contrasts.

    Args:
        contrast_dir (str): Path to the directory with contrast files.
        html_path (str): Path to the HTML file with VIF data.

    Returns:
        List[Dict[str, object]]: A list of input dictionaries with contrast metadata.
    """
    file_list = sorted(glob(f'{contrast_dir}/*'))
    grouped_lists = group_paths_by_filename_pattern(file_list)
    html_vif_df = extract_vif_table(html_path)

    data_type_label = 'Contrast Estimate'
    list_of_input_dicts = []

    for key, value in grouped_lists.items():
        match = re.search(r'contrast_(.*?)_sub', key)
        if not match:
            print(f'Warning: Unexpected key format: {key}')
            continue

        main_title = match.group(1)
        nifti_paths = sorted(value)
        sub_ids = get_subids_from_paths(nifti_paths)
        vif_labels = get_contrast_vif_labels(html_vif_df, sub_ids, main_title)

        if len(sub_ids) != len(vif_labels):
            print(f'Warning: Mismatch in lengths for {main_title}')
            continue

        image_labels = [
            f'{sub_id} {vif_label}' for sub_id, vif_label in zip(sub_ids, vif_labels)
        ]

        list_of_input_dicts.append(
            {
                'main_title': main_title,
                'nifti_paths': nifti_paths,
                'image_labels': image_labels,
                'data_type_label': data_type_label,
            }
        )

    if not list_of_input_dicts:
        print('Warning: No valid input dictionaries created')

    return list_of_input_dicts


def process_contrasts(contrast_dir: str, html_path: str, output_dir: str) -> None:
    """
    Main pipeline for processing contrasts using a VIF table and generating summaries.

    Args:
        contrast_dir (str): Path to contrast files.
        html_path (str): Path to HTML with VIF values.
        output_dir (str): Path where outputs should be saved.
    """
    dicts_list = make_list_of_input_dicts(contrast_dir, html_path)
    dicts_list = dicts_list[:3]  # Limit to 3 for testing or performance
    generate_all_data_summaries(dicts_list, output_dir=output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process contrast files and generate outlier analysis.'
    )
    parser.add_argument(
        '--contrast_dir',
        type=str,
        required=True,
        help='Directory containing contrast files',
    )
    parser.add_argument(
        '--html_path',
        type=str,
        required=True,
        help='Path to the HTML file containing VIF information',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True, help='Directory to save output files'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    process_contrasts(args.contrast_dir, args.html_path, args.output_dir)


if __name__ == '__main__':
    main()

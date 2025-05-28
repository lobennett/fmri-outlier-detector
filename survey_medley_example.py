import argparse
import os
import re
import shutil
from collections import defaultdict
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from plotting_functions import (
    combine_pngs_to_pdf,
    generate_png_files_sd_from_mean,
    summarize_outlier_percentages,
)


def get_subids_from_paths(list_of_paths: List[str]) -> List[str]:
    sub_ids = []
    for path in list_of_paths:
        match = re.search(r'sub[-_]?(\d+)', path)
        if match:
            sub_ids.append(match.group(1))
    return sub_ids


def group_paths_by_filename_pattern(file_paths: List[str]) -> Dict[str, List[str]]:
    grouped = defaultdict(list)
    for path in file_paths:
        filename = os.path.basename(path)
        pattern = re.sub(r'sub[-_]\d+', 'sub_XXX', filename)
        grouped[pattern].append(path)

    # Convert to a regular dictionary and sort the keys
    sorted_grouped = dict(sorted(grouped.items()))
    return sorted_grouped


def extract_vif_table(html_path):
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


def get_contrast_vif_labels(vif_df, sub_labels, contrast_name):
    vif_label = []
    for sub in sub_labels:
        vif_loop = np.round(vif_df.loc[sub, contrast_name]).astype(int)
        vif_label.append(f'(vif={vif_loop})')
    return vif_label


def process_contrasts(contrast_dir: str, html_path: str, output_dir: str):
    file_list = sorted(glob(f'{contrast_dir}/*'))
    grouped_lists = group_paths_by_filename_pattern(file_list)
    temp_output_dir = f'{output_dir}/temp'

    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    os.makedirs(temp_output_dir)

    html_vif_df = extract_vif_table(html_path)

    file_names = []
    outlier_data_all = []
    for key, contrast_paths in grouped_lists.items():
        contrast_name = re.search(r'contrast_(.*?)_sub', key).group(1)
        sub_labels = get_subids_from_paths(contrast_paths)
        vif_labels = get_contrast_vif_labels(html_vif_df, sub_labels, contrast_name)

        file_name, outlier_data = generate_png_files_sd_from_mean(
            sub_labels,
            contrast_paths,
            temp_output_dir,
            contrast_name,
            colorbar_title='Contrast magnitude',
            extra_labels=vif_labels,
        )
        file_names.append(file_name)
        outlier_data_all.append(outlier_data)
    # sort file names
    file_names = sorted(file_names)
    all_img_paths = summarize_outlier_percentages(
        outlier_data_all, output_dir, temp_output_dir
    )
    all_img_paths.extend(file_names)
    combine_pngs_to_pdf(all_img_paths, f'{output_dir}/outlier_analysis.pdf')


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

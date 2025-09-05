import argparse
from pathlib import Path
import numpy as np
import pandas as pd


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
    return parser


def main():
    """
    Main execution pipeline to load, flag, and summarize outlier scans.
    """

    args = create_parser().parse_args()

    thresholds = {
        'vif_strict': args.vif_threshold_strict,
        'outlier_strict': args.outlier_threshold_strict,
        'vif_combined': args.vif_threshold_combined,
        'outlier_combined': args.outlier_threshold_combined,
    }

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
        else:
            print('No scans met the flagging criteria.')

    except FileNotFoundError as e:
        print(f'Error: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()

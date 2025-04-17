import argparse
import gc
import os
import time
from pathlib import Path

import img2pdf  # Make sure to install with: pip install img2pdf
import matplotlib.pyplot as plt
import numpy as np  # Added missing import
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from nilearn import plotting

from utils import (
    all_tasks_fetch_all_contrasts_zstats_vifs,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate contrast heatmap reports')
    parser.add_argument(
        '--data-root',
        type=str,
        help='Root directory for the data',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reports',
        help='Directory to save the output report',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='contrast_heatmaps.pdf',
        help='Filename for the output PDF report',
    )
    return parser.parse_args()


def create_safe_filename(label):
    """Create a safe filename from a contrast label."""
    return label.replace(':', '-').replace('/', '-').replace(' ', '_').replace('*', 'x')


def generate_png_files(contrast_labels, zpaths_vifs_df, output_dir):
    """
    Process contrast labels and generate individual PNG files.

    Args:
        contrast_labels: List of contrast labels to process
        zpaths_vifs_df: DataFrame with paths and data
        output_dir: Directory to save PNG files

    Returns:
        list: List of generated PNG file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    png_files = []

    print(f'Processing {len(contrast_labels)} contrasts')

    for i, con_label in enumerate(contrast_labels):
        print(f'Processing contrast {i + 1}/{len(contrast_labels)}: {con_label}')

        # Filter data for this contrast
        filtered_df = zpaths_vifs_df[
            zpaths_vifs_df['task_contrast'] == con_label
        ].copy()

        # Generate the masked data and plot
        try:
            n_rows = len(filtered_df)
            plot_nrows_ncols = int(np.ceil(np.sqrt(n_rows)))
            fig, axes = plt.subplots(
                plot_nrows_ncols, plot_nrows_ncols, figsize=(15, 15)
            )
            axes = axes.flatten()

            for j, ax in enumerate(axes):
                if j < len(filtered_df):
                    plot_label = filtered_df['subid_ses_vif'].iloc[j]
                    plotting.plot_stat_map(
                        filtered_df['zstat_files'].iloc[j],
                        display_mode='z',
                        cut_coords=[5],
                        colorbar=False,
                        vmin=-4,
                        vmax=4,
                        title=None,
                        axes=ax,
                        bg_img=None,
                    )
                    ax.set_title(plot_label, fontsize=8)
                else:
                    ax.axis('off')  # Hide unused subplots

            cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # Adjust as needed

            # Use the same colormap as plot_stat_map (cold_hot is default)
            cmap = cm.get_cmap('cold_hot')
            norm = Normalize(vmin=-5, vmax=5)

            # Add colorbar
            ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
            fig.suptitle(con_label, fontsize=16)
            plt.subplots_adjust(wspace=0.2, hspace=0.2)

            # Create a safe filename
            safe_label = create_safe_filename(con_label)
            png_path = os.path.join(
                output_dir, f'{i:02d}_{safe_label}_zstat_slice_grid.png'
            )

            # Save as PNG
            fig.savefig(png_path, dpi=100, bbox_inches='tight')
            png_files.append(png_path)

            # Report progress
            print(f'Saved {png_path}')

            # Clean up to free memory
            plt.close(fig)
            del filtered_df, fig, axes  # Fixed variables to delete
            gc.collect()

            # Small pause to ensure memory is released
            time.sleep(0.5)

        except Exception as e:
            print(f'Error processing contrast {con_label}: {str(e)}')
            continue

        # Print progress update every few contrasts
        if (i + 1) % 5 == 0 or (i + 1) == len(contrast_labels):
            print(f'Completed {i + 1}/{len(contrast_labels)} contrasts')

    return png_files


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

    # Sort files by filename to ensure correct order
    sorted_pngs = sorted(png_files)

    print(f'Combining {len(sorted_pngs)} PNG files into PDF: {pdf_path}')

    # Create PDF directory if it doesn't exist
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Convert to bytes for img2pdf
    with open(pdf_path, 'wb') as f:
        f.write(img2pdf.convert(sorted_pngs))

    print(f'PDF created successfully: {pdf_path}')


def main():
    """Main function to run the contrast analysis."""
    args = parse_args()

    print(f'Loading data from {args.data_root}')
    zpaths_vifs_df = all_tasks_fetch_all_contrasts_zstats_vifs(data_root=args.data_root)

    # Get unique contrast labels
    con_labels = zpaths_vifs_df['task_contrast'].unique().tolist()
    print(f'Found {len(con_labels)} unique contrasts')

    # Create paths
    output_dir = Path(args.output_dir)
    png_dir = output_dir / 'png_files'
    pdf_path = output_dir / args.output_file

    # Generate individual PNG files
    png_files = generate_png_files(con_labels, zpaths_vifs_df, png_dir)

    # Combine PNG files into a single PDF
    combine_pngs_to_pdf(png_files, pdf_path)


if __name__ == '__main__':
    main()

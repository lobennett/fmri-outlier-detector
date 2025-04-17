import argparse
import gc
import os
import time
from pathlib import Path

import img2pdf  # Make sure to install with: pip install img2pdf
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.masking import compute_brain_mask

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
    mask_sums = {}  # Track mask sums for verification

    print(f'Processing {len(contrast_labels)} contrasts')

    for i, con_label in enumerate(contrast_labels):
        print(f'Processing contrast {i + 1}/{len(contrast_labels)}: {con_label}')

        # Filter data for this contrast
        filtered_df = zpaths_vifs_df[
            zpaths_vifs_df['task_contrast'] == con_label
        ].copy()

        # Generate the masked data and plot
        try:
            concatenated_img = nib.concat_images(filtered_df['zstat_files'])

            # Print number of files being concatenated
            print(
                f'Concatenating {len(filtered_df["zstat_files"])} files for {con_label}'
            )

            brain_mask = compute_brain_mask(concatenated_img)

            # Calculate and store mask sum for verification
            mask_sum = brain_mask.get_fdata().sum()
            mask_sums[con_label] = mask_sum
            print(f'Mask for {con_label} contains {mask_sum} voxels')

            cut_coords = [-30, -20, -10, 0, 10, 20, 30, 40, 50]
            fig = plt.figure(figsize=(12, 2))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            plotting.plot_roi(
                brain_mask,
                title=None,
                display_mode='z',
                cut_coords=cut_coords,
                axes=ax,
            )
            # Include voxel count in the title for verification
            ax.set_title(f'{con_label} (voxels: {int(mask_sum)})', fontsize=8)

            # Create a safe filename
            safe_label = create_safe_filename(con_label)
            png_path = os.path.join(output_dir, f'{i:02d}_{safe_label}_mask.png')

            # Save as PNG
            fig.savefig(png_path, dpi=100, bbox_inches='tight')
            png_files.append(png_path)

            # Report progress
            print(f'Saved {png_path}')

            # Clean up to free memory
            plt.close(fig)
            del filtered_df, concatenated_img, brain_mask, fig, ax
            gc.collect()

            # Small pause to ensure memory is released
            time.sleep(0.5)

        except Exception as e:
            print(f'Error processing contrast {con_label}: {str(e)}')
            continue

        # Print progress update every few contrasts
        if (i + 1) % 5 == 0 or (i + 1) == len(contrast_labels):
            print(f'Completed {i + 1}/{len(contrast_labels)} contrasts')

    # Print summary of mask sizes
    print('\nMask voxel counts for verification:')
    for label, count in mask_sums.items():
        print(f'{label}: {int(count)} voxels')

    # Check if all masks are identical
    if len(set(mask_sums.values())) == 1:
        print('\nNOTE: All masks contain exactly the same number of voxels.')
    else:
        print('\nMasks contain different numbers of voxels, as expected.')

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

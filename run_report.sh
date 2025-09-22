#!/bin/bash
#SBATCH --job-name=run_report
#SBATCH --output=run_report_%j.out
#SBATCH --error=run_report_%j.err
#SBATCH --time=02:00:00
#SBATCH -p russpold,hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run using uv and specify the venv directory directly
current_dir="$(pwd)"
report_script="/home/users/logben/fmri-outlier-detector/run_report.py"
input_file="/scratch/users/logben/fmriprep_outlier_check_20250912/percent_outlier_data.csv"
level1_dir="/scratch/users/logben/poldrack_glm/level1/output"
output_dir="/scratch/users/logben/fmriprep_outlier_check_20250912"

mkdir -p $output_dir
echo "Running analysis to create figures..."
uv run --directory $current_dir python $report_script \
    --input-file $input_file \
    --level1-dir $level1_dir \
    --output-dir $output_dir
#!/bin/bash
#SBATCH --job-name=make_network
#SBATCH --output=make_network_%j.out
#SBATCH --error=make_network_%j.err
#SBATCH --time=1-00:00:00
#SBATCH -p russpold,hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run using uv and specify the venv directory directly
current_dir="$(pwd)"
network_script="/home/users/logben/fmri-outlier-detector/run_network.py"
# TODO: Change
base_dir="/scratch/users/logben/poldrack_glm/level1/output_test"
output_dir="/scratch/users/logben/fmriprep_outlier_check_20250906"
# TODO: Change
exclusions_file="/home/users/logben/fmri-outlier-detector/exclusions_test.json"

mkdir -p $output_dir
echo "Running analysis to create figures..."
uv run --directory $current_dir python $network_script \
    --base_dir $base_dir \
    --output_dir $output_dir \
    --exclusions-file $exclusions_file

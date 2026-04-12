#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH -t 0-2:00:00
#SBATCH -n 4
#SBATCH --mem=40G
#SBATCH -o /scratch/agurupr6/DynLang-SLAM/results/language_test_%j.out
#SBATCH -e /scratch/agurupr6/DynLang-SLAM/results/language_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agurupr6@asu.edu

module load cuda-12.4.1-gcc-12.1.0
eval "$(conda shell.bash hook)"
conda activate dynlang
cd /scratch/agurupr6/DynLang-SLAM

echo "=== Language Pipeline Test ==="
python scripts/test_language.py

echo ""
echo "=== Language SLAM Test ==="
python scripts/test_language_slam.py

#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-01:00
source venv/bin/activate
python src/train.py

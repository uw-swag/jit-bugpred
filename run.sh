#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=hosseinkeshavarz1997@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=25_train_test.out
#SBATCH --open-mode=append

#module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/venv
#source $SLURM_TMPDIR/venv/bin/activate
#pip install --no-index --upgrade pip
#pip install torch
#pip install --no-index -r requirements.txt

#mkdir $SLURM_TMPDIR/data
#cp -r data/{rawdata.csv,asts_200.json,codebert}
#echo $PWD

source venv/bin/activate
which python
#CUDA_LAUNCH_BLOCKING=1 python -u src/main.py
python -u src/main.py --test

#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=hosseinkeshavarz1997@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=10_train_test.out

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

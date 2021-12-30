#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0-03:00:00

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
python -u src/splitter.py

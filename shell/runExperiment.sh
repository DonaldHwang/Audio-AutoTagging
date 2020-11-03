#!/bin/bash -l

#Request gpu
#SBATCH --gres=gpu:1 --constraint=volta|pascal
#SBATCH --mem-per-cpu 30G
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=8

#Check priorty and current usage of nods
#slurm s gpu,gpushort

##########ssssSBATCH --gres=gpu:v100:1
###########SssssBATCH --gres=gpu:teslap100:1
#""" #SBATCH --gres=gpu:1 --constraint=pascal|volta

module load anaconda3/latest #has cuda 10

source activate /scratch/work/falconr1/envs/audioTagging
cd /scratch/work/falconr1/audioTagging2019

hostname
pwd
python -c 'import librosa'
python -c "import os; print(os.listdir('./data/'))"
python -c "import os; print(os.listdir('./poc/'))"
which python3
echo "init done"

srun python3 ./poc/datasetMelTest.py







#conda install -y -c pytorch torchvision


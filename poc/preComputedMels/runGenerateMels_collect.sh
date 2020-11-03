#!/bin/bash
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=20G

module load anaconda3/latest #has cuda 10
source activate /scratch/work/falconr1/envs/audioTagging

cd /scratch/work/falconr1/audioTagging2019

echo "Yolo"
#Run this before runing the triton stuff!!
#srun python3 ./quickStuff/generateXY_parallel_init.py -m 2

#-s = 0 (training)m -s=1 (validation)
srun python3 ./poc/preComputedMels/generateMels_collect.py -m 2 -s 0 -l 100000 -i 0
#!/bin/bash
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-25

module load anaconda3/latest #has cuda 10
module load sox
source activate audio

cd /scratch/work/falconr1/audioTagging2019

hostname
echo "Yolo"
pwd
which python3

#MTT has about 22000 files

args=(python ./poc/augmentationPaper/datasets/preprocess_audio2npy.py
    -i $SLURM_ARRAY_TASK_ID
    --chunk_size
    1000
    --dataset
    MTTdataset
    --data_limit
    100000
    --resampling_rate
    16000
    --filetype
    npy
    --mono
    1
    )

srun "${args[@]}"

seff $SLURM_JOBID


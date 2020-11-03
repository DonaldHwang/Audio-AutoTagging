#!/bin/bash
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-350


module load anaconda3/latest #has cuda 10
module load sox
source activate audio

cd /scratch/work/falconr1/audioTagging2019

echo "Yolo"
pwd
which python3

srun python3 ./poc/datasetTester/preprocess_audio2npy.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 --mode train --mono False --output_path data/processed/audio_pt --filetype pt
srun python3 ./poc/datasetTester/preprocess_audio2npy.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 --mode validation --mono False --output_path data/processed/audio_pt --filetype pt
srun python3 ./poc/datasetTester/preprocess_audio2npy.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 --mode test --mono False --output_path data/processed/audio_pt --filetype pt

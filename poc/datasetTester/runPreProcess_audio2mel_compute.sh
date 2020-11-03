#!/bin/bash
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-350


module load anaconda3/latest #has cuda 10
module load sox
source activate audio

cd /scratch/work/falconr1/audioTagging2019

echo "Yolo"
pwd
which python3

srun python3 ./poc/datasetTester/preprocess_audio2mel.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 \
    --mode train --mono False --output_path data/processed/spec_npy \
    --resampling_rate 44100 \
    --hop_length_ms 25 \
    --fmin 50 \
    --fmax 10000 \
    --n_mels 96 \
    --n_fft 2048 \
    --use_mels True \
    --max_length 44100

srun python3 ./poc/datasetTester/preprocess_audio2mel.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 \
    --mode validation --mono False --output_path data/processed/spec_npy \
    --resampling_rate 44100 \
    --hop_length_ms 25 \
    --fmin 50 \
    --fmax 10000 \
    --n_mels 96 \
    --n_fft 2048 \
    --use_mels True \
    --max_length 44100

srun python3 ./poc/datasetTester/preprocess_audio2mel.py --platform 2 -i $SLURM_ARRAY_TASK_ID --chunk_size 100 \
    --mode test --mono False --output_path data/processed/spec_npy \
    --resampling_rate 44100 \
    --hop_length_ms 25 \
    --fmin 50 \
    --fmax 10000 \
    --n_mels 96 \
    --n_fft 2048 \
    --use_mels True \
    --max_length 44100
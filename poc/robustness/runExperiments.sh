#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta|pascal
#SBATCH --cpus-per-task=12
#SBATCH --mem 100G
#SBATCH --time 0-1:00:00

module purge
module load anaconda3/latest #has cuda 10
module load sox

#source activate /scratch/work/falconr1/envs/audioTagging
source activate audio

cd /scratch/work/falconr1/audioTagging2019/
cd /scratch/work/falconr1/audioTagging2019/poc/robustness/

hostname
python -c 'import librosa'
python -c "import os; print(os.listdir('./data/'))"
which python
echo "init done"


## Read input parameters
data_limit=$1

echo Running experiments
echo data_limit = $1

#args=(python ./poc/robustness/baseline.py
args=(python ./baseline.py
    --num_workers 44
    --num_epochs 5
    --print_every 50
    --data_limit $data_limit
    --batch_size 32
    --dataset JamendoAudioFolder_npy
    --tforms TorchAudio  # {TorchAudio, Audtorch, MySet}
    --subset top50tags
    --split 0
    --collate default
    --trim_size 1543500  # 44100 * 30 = 1323000, this is 35 secs

    #Parameters for spectrograms
    --resampling_rate 16000
    --hop_length_ms 25
    --fmin 20
    --fmax 8000
    --n_mels 128
    --n_fft 2048
    --use_mels True
    --max_length_frames 1366
    )

#
srun "${args[@]}"



seff $SLURM_JOBID


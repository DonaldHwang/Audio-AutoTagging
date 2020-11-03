#!/usr/bin/env bash
#SBATCH --gres=gpu:1
###SBATCH --constraint=volta|pascal
#SBATCH --cpus-per-task=4
#SBATCH --mem 60G
#SBATCH --time 0-24:00:00
#SBATCH -o "slurm/%j.out"

module purge
module load anaconda3/latest #has cuda 10
module load sox

source activate audio

cd /scratch/work/falconr1/audioTagging2019/poc/augmentationPaper/

#Sanity check, is the anaconda environment loaded ?
hostname
python -c 'import librosa'
which python
echo "init done"

echo Running experiments

args=(python ./train.py
    #--debug_mode
    --job_id $SLURM_JOB_ID
    --tensorboard
    --logging_interval 30

    # Training
    --exp_name '3xx10'
    --batch_size 24         #  64       24 (15 gb 3**10)      16  (28 gb 3**11)
    --num_workers 20
    --num_epochs 250
    --print_every 30
    --clip_grad 3
    --lr 0.01
    --optimizer sgd
    --scheduler noam
    --step_size 5
    --patience 3
    --warmup 20

    # Datasset
    --dataset JamendoAudioDataset
    --dataset_path /m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master
    --prefetch_files
    --data_limit 100000
    --dataset_skip_class_hist

    # Spectrograms
    #--use_time_freq
    --trim_audio_pre 177147  # 59049   177147    531442
    --original_fs 16000
    --new_fs 16000
    --hop_length_ms 40
    --hop_length_ms 20
    --fmin 20
    --fmax 8000
    --n_mels 128
    --n_fft 2048
    --use_mels 1
    --max_length_frames 1366

    # Model
    --model 'SampleCNN'
    --filters_num 128 128 256 256 256 256 256 256 256 512 512 1024  # 3**10

    )


srun "${args[@]}"

conda deactivate

seff $SLURM_JOBID

#    # Datasset
#    --dataset MTTDataset
#    --dataset_path /m/cs/work/falconr1/datasets/MTT/
#    --dataset_tags 15
#    --prefetch_files
#    --data_limit 21000
#
#    # Spectrograms
#    --use_time_freq
#    --original_fs 16000
#    --new_fs 16000
#    --hop_length_ms 40
#    --hop_length_ms 20
#    --fmin 20
#    --fmax 8000
#    --n_mels 128
#    --n_fft 2048
#    --use_mels 1
#    --max_length_frames 1366





#    # Datasset
#    --dataset JamendoAudioDataset
#    --dataset_path /m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master
#    --prefetch_files
#    --data_limit 100000
#    --dataset_skip_class_hist
#
#    # Spectrograms
#    --use_time_freq
#    --original_fs 16000
#    --new_fs 16000
#    --hop_length_ms 40
#    --hop_length_ms 20
#    --fmin 20
#    --fmax 8000
#    --n_mels 128
#    --n_fft 2048
#    --use_mels 1
#    --max_length_frames 1366



#     # Datasset
#    --dataset MNIST
#    --dataset_path /m/cs/work/falconr1/datasets/
#    --prefetch_files
#    --data_limit 100000
#
#    # Spectrograms
#    #--use_time_freq
#    --original_fs 16000
#    --new_fs 16000
#    --hop_length_ms 40
#    --hop_length_ms 20
#    --fmin 20
#    --fmax 8000
#    --n_mels 128
#    --n_fft 2048
#    --use_mels 1
#    --max_length_frames 1366





#    # Datasset
#    --dataset SpeechCommands
#    --dataset_path /m/cs/work/falconr1/datasets/speechCommands/
#    --dataset_tags 20
#    --prefetch_files
#    --data_limit 25000
#
#    # Spectrograms
#    --use_time_freq
#    #--trim_audio_pre 531442  # 59049   177147    531442
#    --original_fs 16000
#    --new_fs 16000
#    --hop_length_ms 40
#    --hop_length_ms 20
#    --fmin 20
#    --fmax 8000
#    --n_mels 128
#    --n_fft 2048
#    --use_mels 1
#    --max_length_frames 1366





    # Model
#    --model 'FCN_legacy'
#    --max_pool 2,4 2,4 2,4 3,5 4,4
#    --filters_num 128 128 128 128 128


    # Model
#    --model 'SampleCNN'
#    --filters_num 128 128 256 256 256 256 256 256 256 256 512 512 1024  # 3**11
#    --filters_num 128 128 256 256 256 256 256 256 256 512 512 1024  # 3**10
#    --filters_num 128 128 128 256 256 256 256 256 512 512 512  # 3**9


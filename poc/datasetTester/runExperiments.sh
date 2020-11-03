#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta|pascal
#SBATCH --cpus-per-task=12
#SBATCH --mem 100G
#SBATCH --time 1-8:00:00

module purge
module load anaconda3/latest #has cuda 10
module load sox

#source activate /scratch/work/falconr1/envs/audioTagging
source activate audio

cd /scratch/work/falconr1/audioTagging2019/

hostname
python -c 'import librosa'
python -c "import os; print(os.listdir('./data/'))"
which python
echo "init done"


## Read input parameters
data_limit=$1

echo Running hypersearch
echo data_limit = $1

#
#
### Spectrograms
#echo "======================= JamendoSpecFolder ==============================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 8 --dataset JamendoSpecFolder --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 16 --dataset JamendoSpecFolder --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoSpecFolder --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoSpecFolder --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 128 --dataset JamendoSpecFolder --collate default
#
#echo "======================= JamendoSpecLMDB ==============================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 8 --dataset JamendoSpecLMDB --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 16 --dataset JamendoSpecLMDB --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoSpecLMDB --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoSpecLMDB --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 128 --dataset JamendoSpecLMDB --collate default
#
#echo "======================= JamendoSpecLMDBsubdir ==============================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 8 --dataset JamendoSpecLMDBsubdir --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 16 --dataset JamendoSpecLMDBsubdir --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoSpecLMDBsubdir --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoSpecLMDBsubdir --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 128 --dataset JamendoSpecLMDBsubdir --collate default
#
#
### Audio
#echo "================= JamendoAudioFolder_npy   collate=default ====================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 8 --dataset JamendoAudioFolder_npy --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 16 --dataset JamendoAudioFolder_npy --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoAudioFolder_npy --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 128 --dataset JamendoAudioFolder_npy --collate default
#
#echo "================= JamendoAudioFolder_npy    collate=seq2seq ========================"
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 8 --dataset JamendoAudioFolder_npy --collate seq2seq
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 16 --dataset JamendoAudioFolder_npy --collate seq2seq
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate seq2seq
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoAudioFolder_npy --collate seq2seq
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 128 --dataset JamendoAudioFolder_npy --collate seq2seq


### Compare Audio and Spec, 10k files:
echo "======================= JamendoSpecFolder ==============================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 64 --dataset JamendoSpecFolder --collate default
#srun python ./poc/datasetTester/testDatasets.py --num_workers 22 --data_limit $data_limit --batch_size 64 --dataset JamendoSpecFolder --collate default

echo "================= JamendoAudioFolder_npy   trim_size ====================="
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 25600
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 51200
#srun python ./poc/datasetTester/testDatasets.py --num_workers 11 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 1323000

#srun python ./poc/datasetTester/testDatasets.py --num_workers 22 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 25600
#srun python ./poc/datasetTester/testDatasets.py --num_workers 22 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 51200
#srun python ./poc/datasetTester/testDatasets.py --num_workers 22 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 1323000

srun python ./poc/datasetTester/testDatasets.py --num_workers 44 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 25600
srun python ./poc/datasetTester/testDatasets.py --num_workers 44 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 51200
srun python ./poc/datasetTester/testDatasets.py --num_workers 44 --data_limit $data_limit --batch_size 32 --dataset JamendoAudioFolder_npy --collate default --trim_size 1323000

seff $SLURM_JOBID


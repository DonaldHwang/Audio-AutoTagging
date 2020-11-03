import os
import argparse

from solver import Solver
from dataset import get_audio_loader


def main(config):
    assert config.mode in {'TRAIN', 'TEST'},\
        'invalid mode: "{}" not in ["TRAIN", "TEST"]'.format(config.mode)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    if config.mode == 'TRAIN':
        data_loader = get_audio_loader(config.audio_path,
                                       config.subset,
                                       config.batch_size,
                                       tr_val = 'train',
                                       split = config.split,
                                       num_workers=config.num_workers)
        valid_loader = get_audio_loader(config.audio_path,
                                        config.subset,
                                        config.batch_size,
                                        tr_val='validation',
                                        split = config.split,
                                        num_workers=config.num_workers)

        solver = Solver(data_loader, valid_loader, config, config.audio_path)

        solver.train()

    elif config.mode == 'TEST':
        data_loader = get_audio_loader(config.audio_path,
                                       config.subset,
                                        config.batch_size,
                                        tr_val = 'test',
                                        split = config.split)

        solver = Solver(data_loader, None, config, config.audio_path)

        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='TRAIN')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--audio_path', type=str, default='/m/cs/work/falconr1/datasets/mtg-jamendo-dataset-master/')
    #parser.add_argument('--audio_path', type=str, default='/Volumes/scratch/work/falconr1/datasets/mtg-jamendo-dataset-master/')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--subset', type=str, default='top50tags')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)

    config = parser.parse_args()

    print(config)
    main(config)

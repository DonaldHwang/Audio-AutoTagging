#!/usr/bin/env python
import argparse
import math
import os
import yaml
from datetime import datetime


def get_result_dir_path(experiment_description: str, root: str = './results'):
    """Returns path where to save training results of a experiment specific result.

    Args:
        root: root path of where to save
        experiment_description: "epoch=50-batch=128-arch=FCN-data=FULL"

    Create the directory, "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    Return directory path(str):
        "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    """
    from datetime import datetime
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H_%M_%S")

    path = f"{experiment_description}__{date_time}"
    path = os.path.join(root, path)
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            print("Path already exists")
        else:
            print(f"Couldn't create {path}.")
            path = root
    else:
        print(f"Save weights to {path}")
    finally:
        return path


def list_of_2d_tuples(s):
    '''
    For the argparser, this reads a string in the format:
    --max_pool 2,2 4,1 3,3
    And returns a list of tuples, as:
    [(2,2), (4,1), (3,3)]
    '''
    try:
        yolo = iter(s.split(' '))
        for coso in yolo:
            x,y = coso.split(',')
            return int(x), int(y)

    except:
        raise argparse.ArgumentTypeError("Tuples must be x,y")


def get_parameters():
    '''
    Gets the parameters for the experiments, including training settings, dataset, spectrograms, and models..
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug_mode', action="store_true",
                        help='Debug mode saves the results to ./results_debug .')
    parser.add_argument('--exp_name', type=str, default='',
                        help="Optional experiment name.")
    parser.add_argument('--job_id', type=str, default='',
                        help='Job id to append to the experiment name. Helps getting the job log.')
    parser.add_argument('--tensorboard', action="store_true", help='Tensorboard logging')
    parser.add_argument('--logging_interval', type=int, default=30,
                        help='Interval of iterations when to log stuff')

    # Parameters for training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help="{'adam', 'sgd'}")
    parser.add_argument('--scheduler', type=str, default='none', help="{'none', 'plateau', 'cyclic', 'noam', steplr}")
    parser.add_argument('--patience', type=int, default=15, help="Patience for the reduce on plateau scheduler. Epochs.")
    parser.add_argument('--step_size', type=int, default=5, help="Step size for the reduce on steplr scheduler. Epochs.")
    parser.add_argument('--clip_grad', type=int, default=0, help="Value to clip gradient norm. Set to 0 to disable.")
    parser.add_argument('--warmup', type=int, default=20, help="Warmup epcohs for the Noam scheduler.")


    parser.add_argument('--print_every', type=int, default=10)


    # Dataset
    parser.add_argument('--dataset', type=str, default='MTTDataset')
    parser.add_argument('--dataset_path', type=str, default='/m/cs/work/falconr1/datasets/MTT/')
    parser.add_argument('--dataset_skip_class_hist', action='store_true',
                        help='Skip the analysis of the dataset and plot class histograms for all subsets.')
    parser.add_argument('--dataset_tags', type=int, default=50,
                        help='Keep the top k classes in the dataset. Currenlty only for MTT.')
    parser.add_argument('--prefetch_files', action='store_true',
                        help='Set to true to try to prefetch all files in the dataset and load them in memory.')
    parser.add_argument('--data_limit', type=int, default=10)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--trim_audio_pre', type=int, default=-1,
                        help="[-1, inf], how many samples to read in audio before transformations, "
                             "when loading the dataset")
    parser.add_argument('--trim_audio_post', type=int, default=-1,
                        help="[-1, inf], how many samples to read in audio before transformations, "
                             "when loading the dataset")

    # Parameters for spectrograms
    spec_group = parser.add_argument_group('spectrogram')
    spec_group.add_argument('--original_fs', type=int, default=16000,
                            help="Original sampling rate.")
    spec_group.add_argument('--new_fs', type=int, default=16000,
                            help="Sampling rate to be used.")
    spec_group.add_argument('--use_time_freq', action='store_true',
                            help="Use time-frequency transforms (spectrograms).")
    spec_group.add_argument('--use_pre_computed_specs', action='store_true',
                            help="Load pre computed spectrograms (some datasets).")
    spec_group.add_argument('--win_length_ms', type=int, default=40,
                            help="Win length in milliseconds.")
    spec_group.add_argument('--hop_length_ms', type=int, default=20,
                            help="Hop length in milliseconds.")
    spec_group.add_argument('--fmin', type=int, default=20,
                            help="Minimum frequency for the mel spectrograms.")
    spec_group.add_argument('--fmax', type=int, default=8000,
                            help="Maximum frequency for the mel spectrograms.")
    spec_group.add_argument('--n_mels', type=int, default=128,
                            help="Number of mel bins.")
    spec_group.add_argument('--n_fft', type=int, default=2048,
                            help="FFT size.")
    spec_group.add_argument('--use_mels', type=int, default=0,
                            help="Use mel filters.")
    spec_group.add_argument('--max_length_frames', type=int, default=128,  ## for the specs
                            help="Desired length in frames for the specs. Positive values will use a random crop of"
                                 "the spectrogram file.")


    model_group = parser.add_argument_group('model')
    model_group.add_argument('--model', type=str, default='FCN_legacy',
                             help='{FCN_legacy, FCN, CRNN, MSCNN, SampleCNN}')
    model_group.add_argument('--max_pool', nargs='+', type=list_of_2d_tuples,
                             help='Sizes for pooling layers.',
                             default=[(2,4), (2,4), (2,4), (3,5), (4,4)])
    model_group.add_argument('--filters_num', nargs='+', type=int,
                             help='Number of filters for the convolution layers.',
                             default=[32, 32, 32, 32, 32])

    # config = parser.parse_args(["--max_pool", "2,2", "2,2", "2,2", "2,2", "3,3", "1,4"])  # test for the list_of_2d_tuples type

    def ms_to_samples(ms, sampling_rate):
        return math.ceil(ms / 1000 * sampling_rate)

    config = parser.parse_args()
    config.hop_length = ms_to_samples(config.hop_length_ms, config.new_fs)
    config.win_length = ms_to_samples(config.win_length_ms, config.new_fs)

    if config.debug_mode:
        config.experiment_description = 'debug_exp'
    else:

        scheduler_type = config.scheduler
        if config.scheduler == 'noam':
            scheduler_param = config.warmup
            param_string = 'warmup'
        elif config.scheduler == 'steplr':
            scheduler_param = config.step_size
            param_string = 'step'
        elif config.scheduler == 'plateau':
            scheduler_param = config.patience
            param_string = 'pat'
        else:
            scheduler_param = ''
        config.experiment_description = "{}__" \
                                        "{}__" \
                                        "{}__" \
                                        "{}__" \
                                        "{}__" \
                                        "lr={}__" \
                                        "{}__" \
                                        "{}={}__" \
                                        "clip={}__" \
                                        "limit={}".format(config.job_id if config.job_id != '' else datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                                                          config.exp_name,
                                                          config.dataset,
                                                          config.model,
                                                          config.optimizer,
                                                          config.lr,
                                                          config.scheduler,
                                                          param_string,
                                                          scheduler_param,
                                                          config.clip_grad,
                                                          config.data_limit,)

    # Results dir saves:
    #   parameters.yaml
    #   Images of selected results, loss, etc
    #   results npy
    if not config.debug_mode:
        config.result_dir = get_result_dir_path(config.experiment_description)
    else:
        config.result_dir = './results_debug/'
        if not os.path.exists(config.result_dir):
            os.mkdir(config.result_dir)

    # Checkpoint saves the model
    config.checkpoint_dir = f'{config.result_dir}/model/'
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    # Logging is for the Tensorboard
    config.logging_dir = './logging/'
    if not os.path.exists(config.logging_dir):
        os.mkdir(config.logging_dir)

    print("")
    print("================ Experiment ================")
    print(config.experiment_description)
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in vars(config).items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    # Save config to disk
    with open(os.path.join(config.result_dir, 'params.yaml'), 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=None)

    return config


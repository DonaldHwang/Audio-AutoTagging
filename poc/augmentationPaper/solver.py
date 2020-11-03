#!/usr/bin/env python
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import time
import os
import utils
from utils import NoamScheduler

from models.FCN import FCN_legacy, init_weights
from models.SampleCNN import flex_sampleCNN

sns.set()


class Solver(object):
    def __init__(self, data_loader_train, data_loader_valid, data_loader_test,
                 config, tensorboard_writer=None, time_freq_transformer=None,
                 model_file=None):
        # Data and configuration parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.data_loader_test = data_loader_test
        self.config = config
        self.writer = tensorboard_writer
        self.transforms = None if time_freq_transformer is None else time_freq_transformer.to(self.device)
        self.model_file = model_file

        self.num_classes = self.data_loader_train.dataset.dataset.num_classes
        self.tags_list = self.data_loader_train.dataset.dataset.tags_list

        # Performance stuff
        torch.backends.cudnn.deterministic = False  # can't be set as true. a pytorch bug makes dilated convs really slow
        torch.backends.cudnn.benchmark = True

        self.build_model()

        summary(self.model, input_size=self._get_input_shape())
        print("")
        print("")

    def _get_input_shape(self):
        tmp_data, _, _ = self.data_loader_train.dataset.dataset[0]
        if self.config.use_time_freq:
            if self.config.use_mels == 0:
                input_size = (1, self.config.n_fft // 2 + 1, self.config.max_length_frames)  # [channels, freq_bins, frames]
            else:
                input_size = (1, self.config.n_mels, self.config.max_length_frames)
        else:
            if len(tmp_data.shape) >= 3:
                input_size = (1, tmp_data.shape[-2], tmp_data.shape[-1])  # precomputed images
            else:
                input_size = (1, tmp_data.shape[-1])  # [channels, timesteps]
        return input_size

    def build_model(self):
        ''' Sets the model and the optimizer. '''

        if self.config.model == 'FCN_legacy':
            self.model = FCN_legacy(input_channels=1,
                                    output_shape=self.num_classes,
                                    filters_num=self.config.filters_num,
                                    max_pool=self.config.max_pool).to(self.device)

        elif self.config.model == 'SampleCNN':
            self.model = flex_sampleCNN(n_filters=self.config.filters_num,
                                        num_classes=self.num_classes,
                                        use_logits=True,
                                        multi_label=True,
                                        dropout=True).to(self.device)

        if self.config.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
        else:
            raise ValueError('Wrong optimizer')

        self.set_scheduler()

        print("")
        print(self.model.__repr__())
        print(self.optimizer)
        print(self.scheduler)

        if self.model_file is not None:
            print("Loading model state from {}".format(self.model_file))
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        else:
            self.model.apply(init_weights)

        return None

    def set_scheduler(self):
        if self.config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                               factor=0.2, patience=self.config.patience, min_lr=1e-6)
        elif self.config.scheduler == 'cyclic':
            self.scheduler = CyclicLR(self.optimizer,
                                      base_lr=0.000001, max_lr=0.1, step_size_up=5, step_size_down=5)
        elif self.config.scheduler == 'noam':
            self.scheduler = NoamScheduler(self.optimizer, self.config.lr, warmup_steps=self.config.warmup)
        elif self.config.scheduler == 'steplr':
            self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size,
                                    gamma=0.8)
        else:
            self.scheduler = None

    def train(self):
        start_t = time.time()
        print(f'Training started at {datetime.now()}')
        print(f'Total number of batches: {len(self.data_loader_train)}')

        best_valid_loss, best_train_epoch_loss, best_roc_auc = 10, 10, 0
        best_step_train_loss, best_step_valid_loss, best_step_valid_roc = 0, 0, 0
        drop_counter = 0
        loss_fn = self.model.loss()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            self.model.train()
            ctr = 0
            for ctr, (audio, target, fname) in enumerate(self.data_loader_train):
                #ctr += 1
                drop_counter += 1
                audio = audio.to(self.device)
                target = target.to(self.device)

                # Time-frequency transform
                if self.transforms is not None:
                    audio = self.transforms(audio)

                # predict
                out = self.model(audio)
                loss = loss_fn(out, target)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.clip_grad > 0:
                    clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                self.optimizer.step()

                epoch_loss += loss.item()

                # print log
                if (ctr) % self.config.print_every == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           epoch + 1, self.config.num_epochs, ctr, len(self.data_loader_train), loss.item(),
                           timedelta(seconds=time.time() - start_t)))

                if self.writer is not None:
                    step = epoch * len(self.data_loader_train) + ctr
                    self.writer.add_scalar('loss', loss.item(), step)
                    self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
                    self.writer.add_scalar('grad_norm', utils.grad_norm(self.model.parameters()), step)

            del audio, target
            epoch_loss = epoch_loss / len(self.data_loader_train)

            # validation
            valid_loss, scores, y_true, y_pred = self._validation(start_t, epoch)
            if self.scheduler is not None:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()

            # Log validation
            if self.writer is not None:
                step = epoch * len(self.data_loader_train) + ctr
                self.writer.add_scalar('valid_loss', valid_loss, step)
                self.writer.add_scalar('valid_roc_auc_macro', scores['roc_auc_macro'], step)
                if not self.config.debug_mode:
                    self.writer.add_figure('valid_class', utils.compare_predictions(y_true, y_pred, filepath=None), step)

            # Save model, with respect to validation loss
            if valid_loss < best_valid_loss:
                # print('best model: %4f' % valid_loss)
                best_step_valid_loss = drop_counter
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'best_model_valid_loss.pth'))

            # Save model, with respect to validation roc_auc
            if scores['roc_auc_macro'] > best_roc_auc:
                best_step_valid_roc = drop_counter
                best_roc_auc = scores['roc_auc_macro']
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'best_model_valid_roc.pth'))

            # Save best model according to training loss
            if epoch_loss < best_train_epoch_loss:
                best_step_train_loss = drop_counter
                best_train_epoch_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'best_model_train.pth'))

        print("{} Training finished. -----------------------  Elapsed: {}".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            timedelta(seconds=time.time() - start_t))
        )
        print("Best step (validation loss) = {} . ".format(best_step_valid_loss))
        print("Best step (validation roc_auc) = {} .".format(best_step_valid_roc))
        print("Best step (training loss) = {} .".format(best_step_train_loss))

        # Save last model
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'best_model_final.pth'))

    def _validation(self, start_t, epoch):
        pred_scores = []
        true_class = []
        valid_loss = 0

        loss_fn = self.model.loss()
        self.model.eval()
        with torch.no_grad():
            for _, (audio, target, fname) in enumerate(self.data_loader_valid):
                audio = audio.to(self.device)
                target = target.to(self.device)

                # Time-frequency transform
                if self.transforms is not None:
                    audio = self.transforms(audio)

                # Predict
                out = self.model(audio)
                loss = loss_fn(out, target)
                valid_loss += loss.item()

                # Append results for evaluation
                out = self.model.forward_with_evaluation(audio)
                out = out.detach().cpu()
                target = target.detach().cpu()
                for prd in out:
                    pred_scores.append(list(np.array(prd)))
                for gt in target:
                    true_class.append(list(np.array(gt)))

        valid_loss = valid_loss / len(self.data_loader_valid)

        # print log
        print("[%s] Epoch [%d/%d], validation loss: %.4f Elapsed: %s" %
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               epoch + 1, self.config.num_epochs, valid_loss,
               timedelta(seconds=time.time() - start_t)))

        true_class = np.array(true_class)
        pred_scores = np.array(pred_scores)

        scores = utils.compute_evaluation_metrics(np.copy(true_class), np.copy(pred_scores), self.tags_list,
                                                  filepath=self.config.result_dir if not self.config.debug_mode else None,
                                                  class_threshold=0.5, multilabel=True, verbose=False)

        return valid_loss, scores, true_class, pred_scores

    def test(self):
        pred_scores = []
        true_class = []
        test_loss = 0

        loss_fn = self.model.loss()
        self.model.eval()
        with torch.no_grad():
            for _, (audio, target, fname) in enumerate(self.data_loader_test):
                audio = audio.to(self.device)
                target = target.to(self.device)

                # Time-frequency transform
                if self.transforms is not None:
                    audio = self.transforms(audio)

                # Predict
                out = self.model(audio)
                loss = loss_fn(out, target)
                test_loss += loss.item()

                # Append results for evaluation
                del out
                torch.cuda.empty_cache()
                out = self.model.forward_with_evaluation(audio)
                out = out.detach().cpu()
                target = target.detach().cpu()
                for prd in out:
                    pred_scores.append(list(np.array(prd)))
                for gt in target:
                    true_class.append(list(np.array(gt)))

        test_loss = test_loss / len(self.data_loader_test)

        # print log
        print("[%s] TESTING.  Test loss %.4f" %
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               test_loss))

        true_class = np.array(true_class)
        pred_scores = np.array(pred_scores)

        scores = utils.compute_evaluation_metrics(np.copy(true_class), np.copy(pred_scores), self.tags_list, filepath=self.config.result_dir,
                                                  class_threshold=0.5, multilabel=True)
        return scores, true_class, pred_scores


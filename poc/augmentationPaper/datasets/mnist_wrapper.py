import torchvision.datasets as vis_datasets
import torch.utils.data


class MYMNIST(vis_datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MYMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

    def __getitem__(self, item):
        img, target = super().__getitem__(item)

        return img, target, 'fake'

    @property
    def num_classes(self):
        return len(super().classes)

    @property
    def tags_list(self):
        return super().classes
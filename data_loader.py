import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label


def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    # trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    loadedZeros = torch.load('./data/zeroImagesTrain.pt')
    loadedOnes = torch.load('./data/oneImagesTrain.pt')

    dataset01 = torch.cat((loadedZeros, loadedOnes), dim=0)
    labels01 = torch.zeros(len(loadedZeros)+len(loadedOnes))
    labels01[len(loadedOnes):] = 1

    torch.manual_seed(52)
    indices = torch.randperm(dataset01.size(0))

    data_array = dataset01[indices]
    label_array = labels01[indices].long()

    num_train = len(dataset01)
    split = int(np.floor(valid_size * num_train))

    trainDataset = CustomDataset(data_array[split:], label_array[split:])
    validDataset = CustomDataset(data_array[:split], label_array[:split])

    # can simplify this considerably...kist shuffle and partition train, valid
    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    valid_loader = torch.utils.data.DataLoader(
        validDataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    # load dataset
    loadedZeros = torch.load('./data/zeroImagesTest.pt')
    loadedOnes = torch.load('./data/oneImagesTest.pt')

    dataset01 = torch.cat((loadedZeros, loadedOnes), dim=0)
    labels01 = torch.zeros(len(loadedZeros)+len(loadedOnes))
    labels01[len(loadedOnes):] = 1

    torch.manual_seed(52)
    indices = torch.randperm(dataset01.size(0))

    data_array = dataset01[indices]
    label_array = labels01[indices].long()


#    Create an instance of your custom dataset
    dataset = CustomDataset(data_array, label_array)
    print(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return data_loader

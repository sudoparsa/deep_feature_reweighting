import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

class DominoeMnistCifarDataset(Dataset):
    def __init__(self, data_type, spuriousity):

        assert data_type in ['train', 'val', 'test'], print("Error! no data_type found")
        assert not data_type in ['val', 'test'] or spuriousity == 0.5, print("Error! val and test must have spuriousity=0.5")
        self.spuriousity = spuriousity
        self.data_type = data_type

        cifar_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
        ])

        mnist_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=0.485,
                                           std=0.229),
        ])

        mnist_train_raw = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
        cifar_train_raw = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=cifar_transform)
        mnist_train, mnist_valid = random_split(mnist_train_raw, [0.80, 0.20], generator=torch.Generator().manual_seed(42))
        cifar_train, cifar_valid = random_split(cifar_train_raw, [0.80, 0.20], generator=torch.Generator().manual_seed(42))

        mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transform)
        cifar_test = torchvision.datasets.CIFAR10('./data/FashionMNIST/', train=False, download=True, transform=cifar_transform)

        mnist_dataset = None
        cifar_dataset = None
        if data_type == 'train':
            mnist_dataset = mnist_train
            cifar_dataset = cifar_train
        elif data_type == 'val':
            mnist_dataset = mnist_valid
            cifar_dataset = cifar_valid
            spuriousity = 0.5
        elif data_type == 'test':
            mnist_dataset = mnist_test
            cifar_dataset = cifar_test


        x, y, g = make_spurious_dataset(mnist_dataset, cifar_dataset, spuriousity)
        self.x = x
        self.y = y
        self.g = g


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        y = 1 if self.y[idx][0] == 1 else 0
        g = np.where(self.g[idx] == 1)[0][0]
        return self.x[idx], y, g, -1


def keep_only_lbls(dataset, lbls):
  lbls = {lbl: i for i, lbl in enumerate(lbls)}
  final_X, final_Y = [], []
  for x, y in dataset:
    if y in lbls:
      final_X.append(x)
      final_Y.append(lbls[y])
  X = torch.stack(final_X)
  Y = torch.tensor(final_Y).float().view(-1,1)
  return X, Y

def format_mnist(imgs):
  imgs = np.stack([np.pad(imgs[i][0], 2, constant_values=0)[None,:] for i in range(len(imgs))])
  imgs = np.repeat(imgs, 3, axis=1)
  return torch.tensor(imgs)

def make_spurious_dataset(mnist_dataset, cifar_dataset, spuriousity):
    X_m_0, _ = keep_only_lbls(mnist_dataset, lbls=[0])
    X_m_1, _ = keep_only_lbls(mnist_dataset, lbls=[1])
    X_m_0 = format_mnist(X_m_0.view(-1, 1, 28, 28))
    X_m_1 = format_mnist(X_m_1.view(-1, 1, 28, 28))
    X_m_0 = X_m_0[torch.randperm(len(X_m_0))]
    X_m_1 = X_m_1[torch.randperm(len(X_m_1))]

    X_c_1, _ = keep_only_lbls(cifar_dataset, lbls=[1])
    X_c_9, _ = keep_only_lbls(cifar_dataset, lbls=[9])
    X_c_1 = X_c_1[torch.randperm(len(X_c_1))]
    X_c_9 = X_c_9[torch.randperm(len(X_c_9))]

    min_length = min(len(X_m_0), len(X_m_1), len(X_c_1), len(X_c_9))
    X_m_0, _ = random_split(X_m_0, [min_length, len(X_m_0) - min_length], generator=torch.Generator().manual_seed(42))
    X_m_1, _ = random_split(X_m_1, [min_length, len(X_m_1) - min_length], generator=torch.Generator().manual_seed(42))
    X_c_1, _ = random_split(X_c_1, [min_length, len(X_c_1) - min_length], generator=torch.Generator().manual_seed(42))
    X_c_9, _ = random_split(X_c_9, [min_length, len(X_c_9) - min_length], generator=torch.Generator().manual_seed(42))

    X_m_0_maj, X_m_0_min = random_split(X_m_0, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))
    X_m_1_maj, X_m_1_min = random_split(X_m_1, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))

    X_c_1_maj, X_c_1_min = random_split(X_c_1, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))
    X_c_9_maj, X_c_9_min = random_split(X_c_9, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))

    group_0_X = torch.cat((X_c_1_maj[:], X_m_0_maj[:]), dim=2)
    group_0_Y = torch.zeros(len(group_0_X))
    group_0_G = torch.tensor([0] * len(group_0_X))

    group_1_X = torch.cat((X_c_1_min[:], X_m_1_min[:]), dim=2)
    group_1_Y = torch.zeros(len(group_1_X))
    group_1_G = torch.tensor([1] * len(group_1_X))

    group_2_X = torch.cat((X_c_9_min[:], X_m_0_min[:]), dim=2)
    group_2_Y = torch.ones(len(group_2_X))
    group_2_G = torch.tensor([2] * len(group_2_X))

    group_3_X = torch.cat((X_c_9_maj[:], X_m_1_maj[:]), dim=2)
    group_3_Y = torch.ones(len(group_3_X))
    group_3_G = torch.tensor([3] * len(group_3_X))

    total_x = torch.cat((group_0_X, group_1_X, group_2_X, group_3_X))
    total_y = torch.cat((group_0_Y, group_1_Y, group_2_Y, group_3_Y))
    total_g = torch.cat((group_0_G, group_1_G, group_2_G, group_3_G))

    return total_x, total_y, total_g

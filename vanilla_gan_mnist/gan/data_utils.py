import torch
import torch.utils.data
import torchvision.transforms
import torchvision.datasets
from torch.autograd import Variable


def mnist_loader(batch_size: int = 32, num_workers: int = 0, root: str = "."):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mnist_train_set = torchvision.datasets.MNIST(root=root, train=True, transform=transforms, download=True)
    return torch.utils.data.DataLoader(
        dataset=mnist_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def random_noise(batch_size: int, noise_dim: int) -> Variable:
    return Variable(torch.randn(batch_size, noise_dim))


def ones(batch_size: int, vector_dim: int) -> Variable:
    return Variable(torch.ones(batch_size, vector_dim))


def zeros(batch_size: int, vector_dim: int) -> Variable:
    return Variable(torch.zeros(batch_size, vector_dim))


def flatten(x: Variable) -> Variable:
    s = 1
    for n in x.shape[1:]:
        s *= n
    return x.view(x.shape[0], s)

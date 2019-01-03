import os
from typing import List

import torch
from torchvision.utils import save_image
from tqdm import tqdm


from .models import Generator
from .training import sample_generator


def test(generator: Generator, samples_save_to: str, n_samples: int, sample_shape: List, device: torch.device):

    os.makedirs(samples_save_to, exist_ok=True)

    generator = generator.to(device).eval()
    for i in tqdm(range(n_samples)):
        save_image(sample_generator(generator, 1, sample_shape, device), os.path.join(samples_save_to, f'{i}.jpg'))

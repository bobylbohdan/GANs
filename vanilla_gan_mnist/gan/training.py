import os
from typing import Dict, Iterator, List
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


from .models import Discriminator, Generator
from .data_utils import ones, zeros, random_noise, flatten
from .utils import save_checkpoint


def make_optimization_step(optimizer: Optimizer, loss: nn.Module):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_discriminator(discriminator: Discriminator, generator: Generator, optimizer: Optimizer,
                        loss_criterion: nn.Module, data_iter: Iterator, discriminator_train_steps_per_batch: int,
                        device: torch.device) -> float:
        discriminator.train()
        generator.eval()
        cumulative_loss = 0.
        for _ in range(discriminator_train_steps_per_batch):
            real_x = flatten(next(data_iter)[0]).to(device)
            real_y = ones(real_x.shape[0], 1).to(device)

            fake_x = generator.forward(random_noise(real_x.shape[0], generator.input_dim).to(device)).detach().to(device)
            fake_y = zeros(fake_x.shape[0], 1).to(device)

            y_real_pred = discriminator.forward(real_x).to(device)
            y_fake_pred = discriminator.forward(fake_x).to(device)

            loss = (loss_criterion(y_real_pred, real_y) + loss_criterion(y_fake_pred, fake_y)) / 2

            make_optimization_step(optimizer, loss)
            cumulative_loss += loss.cpu().data.numpy()

        return cumulative_loss / discriminator_train_steps_per_batch


def train_generator(discriminator: Discriminator, generator: Generator, optimizer: Optimizer, loss_criterion: nn.Module,
                    batch_size: int, generator_train_steps_per_batch: int, device: torch.device) -> float:
    discriminator.eval()
    generator.train()
    cumulative_loss = 0.
    for _ in range(generator_train_steps_per_batch):
        fake_x = generator.forward(random_noise(batch_size, generator.input_dim).to(device)).to(device)
        fake_y = ones(fake_x.shape[0], 1).to(device)

        y_fake_pred = discriminator.forward(fake_x).to(device)

        loss = loss_criterion(y_fake_pred, fake_y)

        make_optimization_step(optimizer, loss)
        cumulative_loss += loss.cpu().data.numpy()

    return cumulative_loss / generator_train_steps_per_batch


def sample_generator(generator: Generator, n_samples: int, reshape_size: List, device: torch.device):
    generated = generator.forward(random_noise(n_samples, generator.input_dim).to(device))
    return make_grid(generated.view(n_samples, *reshape_size).cpu(), normalize=True, scale_each=True)


def train(discriminator: Discriminator, generator: Generator, dataloader: DataLoader, optimizers: Dict[str, Optimizer],
          n_train_batches: int, discriminator_train_steps_per_batch: int, generator_train_steps_per_batch: int,
          device: torch.device, writer: SummaryWriter, logging_and_sampling_config: Dict, checkpoint_config: Dict):

    os.makedirs(os.path.join(checkpoint_config['base_path'], checkpoint_config['discriminator_path']), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_config['base_path'], checkpoint_config['generator_path']), exist_ok=True)

    discriminator = discriminator.to(device)
    generator = generator.to(device)

    d_optimizer = optimizers['discriminator']
    g_optimizer = optimizers['generator']

    loss_criterion = nn.BCELoss().to(device)

    data_iter = cycle(dataloader)
    for batch_num in tqdm(range(1, n_train_batches + 1), total=n_train_batches, desc="Processed batches: "):
        d_avg_loss_n_batches = train_discriminator(discriminator, generator, d_optimizer, loss_criterion, data_iter,
                                                   discriminator_train_steps_per_batch, device)
        g_avg_loss_n_batches = train_generator(discriminator, generator, g_optimizer, loss_criterion,
                                               dataloader.batch_size, generator_train_steps_per_batch, device)

        if batch_num % logging_and_sampling_config['log_freq'] == 0:
            writer.add_scalar("discriminator_loss", d_avg_loss_n_batches, batch_num)
            writer.add_scalar("generator_loss", g_avg_loss_n_batches, batch_num)

        if batch_num % logging_and_sampling_config['sample_freq'] == 0:
            sample_shape = [
                logging_and_sampling_config['sample_shape']['channels'],
                logging_and_sampling_config['sample_shape']['width'],
                logging_and_sampling_config['sample_shape']['height']
            ]

            samples = sample_generator(generator, logging_and_sampling_config['n_samples'], sample_shape, device)
            writer.add_image("generator_samples", samples, batch_num)

        if batch_num % checkpoint_config['frequency'] == 0:
            d_path = os.path.join(
                checkpoint_config['base_path'], checkpoint_config['discriminator_path'], f'batch_{batch_num}.pt')
            g_path = os.path.join(
                checkpoint_config['base_path'], checkpoint_config['generator_path'], f'batch_{batch_num}.pt')
            save_checkpoint(discriminator, d_path)
            save_checkpoint(generator, g_path)

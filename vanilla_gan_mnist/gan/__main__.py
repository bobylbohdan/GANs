from pprint import pprint
from typing import Dict

import click
from tensorboardX import SummaryWriter

from gan.models import Discriminator, Generator
from gan.utils import load_yaml_config, get_device, get_optimizer, load_checkpoint
from gan.data_utils import mnist_loader
from gan.training import train
from gan.testing import test


def train_mode(config: Dict):
    device = get_device()

    pprint(config)
    print(f"Device: {device}")

    discriminator = Discriminator()
    load_checkpoint(discriminator, config['initialization']['discriminator'])
    generator = Generator()
    load_checkpoint(discriminator, config['initialization']['generator'])
    optimizers = {
        "discriminator": get_optimizer(discriminator, config['optimizers']['discriminator']),
        "generator": get_optimizer(generator, config['optimizers']['generator'])
    }
    dataloader = mnist_loader(batch_size=config['general']['batch_size'],
                              num_workers=config['general']['n_dataloader_workers'],
                              root=config['general']['dataset_path'])

    writer = SummaryWriter()
    train(discriminator=discriminator, generator=generator, dataloader=dataloader, optimizers=optimizers,
          n_train_batches=config['general']['n_batches'],
          discriminator_train_steps_per_batch=config['general']['discriminator_steps_per_batch'],
          generator_train_steps_per_batch=config['general']['generator_steps_per_batch'], device=device,
          writer=writer, logging_and_sampling_config=config['logging'],
          checkpoint_config=config['checkpoints'])


def test_mode(config: Dict):
    device = get_device()

    pprint(config)
    print(f"Device: {device}")

    generator = Generator()
    load_checkpoint(generator, config['generator_checkpoint'])

    sample_shape = [
        config['sample_shape']['channels'],
        config['sample_shape']['width'],
        config['sample_shape']['height']
    ]
    test(generator=generator, samples_save_to=config['samples_save_to'], n_samples=config['n_samples'],
         sample_shape=sample_shape, device=device)


@click.command()
@click.option("--config_path")
def main(config_path: str):
    config = load_yaml_config(config_path)
    if config['mode'] == 'train':
        train_mode(config)
    elif config['mode'] == 'test':
        test_mode(config)
    else:
        raise ValueError(f"There has to be 'mode' field with 'train' or 'test' value in {config_path}")


if __name__ == "__main__":
    main()

from typing import Callable, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def add_random_numbers_to_borders(images, n) -> torch.Tensor:
    """
    Add random numbers to specified number of areas in the borders of the images.

    Args:
    - images (torch.Tensor): Tensor of images of size (N, 1, 32, 32).
    - n (int): Number of areas (ranging from 0 to 8) to add random numbers to.
    - seed (int): Seed for random number generation.

    Returns:
    - torch.Tensor: Tensor of images with random numbers added to specified areas.
    """

    batch_size = images.size(0)
    img_size = images.size(-1)

    # Generate masks for each area
    masks = torch.zeros(batch_size, 1, img_size, img_size)

    # Divide the borders into 8 areas
    corner_size = 8
    border_size = 16

    if n > 0:
        # Deterministically select n areas to modify
        areas_to_modify = np.arange(8)[:n]

        for i, area in enumerate(areas_to_modify):
            if area < 4:  # Corners
                if area == 0:
                    masks[:, :, :corner_size, :corner_size] = i + 1
                elif area == 1:
                    masks[:, :, :corner_size, -corner_size:] = i + 1
                elif area == 2:
                    masks[:, :, -corner_size:, :corner_size] = i + 1
                else:
                    masks[:, :, -corner_size:, -corner_size:] = i + 1
            else:  # Borders between corners
                if area == 4:
                    masks[:, :, :corner_size,
                          corner_size:img_size-corner_size] = i + 1
                elif area == 5:
                    masks[:, :, corner_size:img_size -
                          corner_size, :corner_size] = i + 1
                elif area == 6:
                    masks[:, :, -corner_size:,
                          corner_size:img_size-corner_size] = i + 1
                else:
                    masks[:, :, corner_size:img_size -
                          corner_size, -corner_size:] = i + 1

    # Generate random numbers from normal distribution for each area
    random_numbers = 2. * torch.rand(batch_size, n) - 1.

    # Apply masks to images
    masked_images = images.clone()  # Create a copy to preserve original images
    for i in range(batch_size):
        for j, area in enumerate(areas_to_modify):
            masked_images[i] += (masks[i] == (j + 1)
                                 ).float() * random_numbers[i, j]

    return masked_images


def _generate_scaled_dataset(scaled_size: int, dataset_name="FMNIST") -> Tuple[np.ndarray, np.ndarray]:

    size = int(1e6)
    train = _download_base_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=size, shuffle=False)

    data, labels = next(iter(train_loader))

    data_scaled = F.interpolate(
        data, size=16, mode='bilinear', align_corners=False)

    if scaled_size  != 16:
        data_scaled = F.interpolate(
            data_scaled, size=scaled_size, mode='bilinear', align_corners=False)
    return data_scaled.numpy(), labels.numpy()


def generate_downscaled_dataset(dataset_name="FMNIST"):
    return _generate_scaled_dataset(scaled_size=16, dataset_name=dataset_name)


def generate_upscaled_dataset(dataset_name="FMNIST"):
    return _generate_scaled_dataset(scaled_size=32, dataset_name=dataset_name)


def generate_stretched_dataset(dataset_name="FMNIST", power=4) -> tuple[np.ndarray, np.ndarray]:
    dataset, labels = _generate_scaled_dataset(scaled_size=16, dataset_name=dataset_name)
    data_min = dataset.min()
    data_max = dataset.max()
    dataset = (dataset - data_min) / (data_max - data_min)
    return dataset**power, labels


def generate_padded_dataset(dataset_name="FMNIST", additional_dimensions=0):
    assert (additional_dimensions <= 8) & (additional_dimensions >= 0)

    data_downscaled, labels = _generate_scaled_dataset(
        scaled_size=16, dataset_name=dataset_name)
    data_padded = F.pad(torch.from_numpy(data_downscaled),
                        (8, 8, 8, 8), mode='reflect')

    if additional_dimensions > 0:
        data_padded = add_random_numbers_to_borders(
            data_padded, additional_dimensions)

    return data_padded.numpy(), labels


def _download_base_dataset(
    type: str,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
):
    if type == "FMNIST":
        data = datasets.FashionMNIST(
            root='./data/base_datasets', train=True, download=True, transform=transform)
    elif type == "MNIST":
        data = datasets.MNIST(root='./data/base_datasets', train=True,
                              download=True, transform=transform)

    return data


def generate_sampled_dataset(
        sampling_step: int,
        N_train: int,
        type="FMNIST"
) -> tuple[np.ndarray, np.ndarray]:

    size = int(1e6)
    train_val_test = _download_base_dataset(
        type,
        transform=transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    )

    train_loader = torch.utils.data.DataLoader(
        train_val_test,
        batch_size=size,
        shuffle=False)

    all_data, all_labels = next(iter(train_loader))

    train_data = all_data[:N_train]
    train_labels = all_labels[:N_train]

    for _ in range(1, sampling_step):
        try:
            train_data, _, train_labels, _ = train_test_split(
                train_data,
                train_labels,
                test_size=0.5,
                stratify=train_labels,
                random_state=0
            )
        except ValueError:
            print("Splitting ends here.")
            raise StopIteration

    sampled_data = torch.vstack([train_data, all_data[N_train:]]).numpy()
    sampled_labels = torch.concat([train_labels, all_labels[N_train:]]).numpy()

    return sampled_data, sampled_labels

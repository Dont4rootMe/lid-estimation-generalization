# PCA
from pathlib import Path
import joblib
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def train_or_load_pca(
        dataset_name="fmnist",
        image_size=32,
        pca_desired_class=7,
        pca_n_components=20,
        seed=0,
        base_dataset_root_dir="data/base_datasets",
        pca_output_dir="./",
        pca_load_from_disk=False,
):
    Path(pca_output_dir).mkdir(exist_ok=True, parents=True)

    if pca_load_from_disk:
        trained_pca = joblib.load(Path(pca_output_dir) / f"pca.joblib")
    else:
        assert image_size == 32, "image_size other than 32 is not supported right now!"

        data_for_pca = get_image_data(
            type=dataset_name,
            desired_class=pca_desired_class,
            data_root=base_dataset_root_dir,
        )

        trained_pca = train_pca(
            data_for_pca,
            n_components=pca_n_components,
            seed=seed,
        )
        joblib.dump(trained_pca, Path(pca_output_dir) / f"pca.joblib")

    return trained_pca


def pairwise_euclidean_distances(dataset):
    distances_squared = np.sum(
        (dataset[:, np.newaxis] - dataset) ** 2, axis=-1)
    distances = np.sqrt(distances_squared)
    return distances


def plot_mosaic(images, num_cols=10):
    # TODO: untested in prod
    num_images = len(images)
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 1.5 * num_rows))
    axs = axs.flatten()

    for i in range(num_images):
        # Assuming images are flattened to 1D
        image = images[i].reshape(28, 28)
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')

    # Remove empty subplots if necessary
    for j in range(num_images, num_rows * num_cols):
        fig.delaxes(axs[j])
    plt.show()
    plt.plot(images.min(axis=0), label='min')
    plt.plot(images.max(axis=0), label='max')
    plt.plot(images.max(axis=0) - images.min(axis=0), label='diff')
    plt.legend()
    plt.show()


def get_image_data(type="fmnist", desired_class=9, size=int(1e6), data_root="./data/fmnist"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if type == "fmnist":
        train = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=transform)

    else:
        raise Exception("Only FMNIST is implemented for PCA now.")
    train_filtered = [item for item in train if item[1] == desired_class]
    train_loader = torch.utils.data.DataLoader(
        train_filtered, batch_size=size, shuffle=False)
    data, labels = next(iter(train_loader))
    return data


def train_pca(data, n_components=30, seed=0):
    data = data.view(data.size(0), -1)
    pca = PCA(n_components=n_components, random_state=seed)
    pca_result = pca.fit_transform(data.numpy())
    return pca


def generate_points_on_sphere(N, dim=None):
    assert dim is not None
    x = np.random.normal(size=(N, dim))
    lam = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    x = x / lam
    return x


def prepare_pca_dataset(coefficients, pca, target_shape=(-1, 1, 28, 28)):
    pca_dataset = np.dot(coefficients, pca.components_) + pca.mean_
    return pca_dataset.reshape(target_shape)


def uniform_pca(N, dim=2, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    for i in range(dim):
        coefficients[:, i] = np.random.uniform(high=3, low=-3, size=N)
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * dim
    return dataset, lid, coefficients


def gaussian_pca(N, dim=2, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    for i in range(dim):
        coefficients[:, i] = np.random.normal(size=N)
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * dim
    return dataset, lid, coefficients


def sphere_pca(N, dim=2, pca=None, radius=1., seed=0):
    np.random.seed(seed)
    assert pca is not None
    assert dim > 1
    coefficients = np.zeros((N, pca.n_components))
    sphere = generate_points_on_sphere(N, dim)
    coefficients[:, :dim] = sphere * radius
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * (dim - 1)
    return dataset, lid, coefficients


def crescent_moon_pca(N, pca=None, radius=1., seed=0):
    np.random.seed(seed)
    assert pca is not None
    x = np.random.uniform(-1, 1, size=(20*N, 3))
    mask = ((x[:, 0]**2 + x[:, 1]**2) <=
            1) & (np.sqrt(x[:, 0]**2 + (x[:, 1] - 0.1)**2) >= 0.899)
    x = radius * x[mask]
    assert len(x) >= N
    coefficients = np.zeros((N, pca.n_components))
    coefficients[:, :3] = x[:N]
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * 3
    return dataset, lid, coefficients


def exp_pca(N, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    x = np.random.uniform(0, 8, size=(N, 3))
    r = 3 * np.exp(-x[:, 0])
    theta = np.random.uniform(low=0, high=2*np.pi, size=N)
    x[:, 0] -= 4
    x[:, 1] = r * np.sin(theta)
    x[:, 2] = r * np.cos(theta)
    coefficients[:, :3] = x
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * 2
    return dataset, lid, coefficients


def spiral_pca(N, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    t = np.random.uniform(1, 100, N)
    r = 1 / t
    x = r * np.sin(1 / r * t)
    y = r * np.cos(1 / r * t)
    coefficients[:, 0] = x
    coefficients[:, 1] = y
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int")
    return dataset, lid, coefficients


def sphere4_pca(N, dim=6, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    assert dim > 1
    coefficients = np.zeros((N, pca.n_components))
    for i in range(N):
        direction = np.random.randint(4)
        scale = 3 / 3**direction
        sample = (generate_points_on_sphere(1, dim) * scale).reshape(dim)
        if direction == 0:
            sample[0] = sample[0] + 3
            sample[1] = sample[1] + 3
        elif direction == 1:
            sample[0] = sample[0] - 3
            sample[1] = sample[1] - 3
        elif direction == 2:
            sample[0] = sample[0] + 3
            sample[1] = sample[1] - 3
        elif direction == 3:
            sample[0] = sample[0] - 3
            sample[1] = sample[1] + 3
        coefficients[i, :dim] = sample
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * (dim - 1)
    return dataset, lid, coefficients


def gaussian4_pca(N, dim=5, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    for i in range(N):
        direction = np.random.randint(4)
        # direction = np.random.choice(choices, p=probabilities)
        scale = 1 / 3**direction
        sample = np.random.normal(scale=scale, size=(dim))
        if direction == 0:
            sample[0] = sample[0] + 3
            sample[1] = sample[1] + 3
        elif direction == 1:
            sample[0] = sample[0] - 3
            sample[1] = sample[1] - 3
        elif direction == 2:
            sample[0] = sample[0] + 3
            sample[1] = sample[1] - 3
        elif direction == 3:
            sample[0] = sample[0] - 3
            sample[1] = sample[1] + 3
        coefficients[i, :dim] = sample
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * dim
    return dataset, lid, coefficients


def spaghetti_pca(N, dim=20, pca=None, seed=0):
    np.random.seed(seed)
    assert pca is not None
    coefficients = np.zeros((N, pca.n_components))
    theta = np.random.uniform(0, 2*np.pi, N)
    for i in range(dim):
        coefficients[:,i] = 1 * np.sin((i+2)*theta)
    dataset = prepare_pca_dataset(coefficients, pca)
    lid = np.ones(N).astype("int") * dim
    return dataset, lid, coefficients

import os
from pathlib import Path
import shutil
import tarfile
from typing import Optional

import requests
from tqdm import tqdm
from generators.utils.arrows import make_arrows_dataset
from generators.utils.padded_and_downscaled import generate_downscaled_dataset, generate_padded_dataset, generate_sampled_dataset, generate_stretched_dataset, generate_upscaled_dataset
from generators import LIDBenchmarkDatasetGenerator, PCALIDBenchmarkDatasetGenerator
from generators.utils.pca import crescent_moon_pca, exp_pca, gaussian4_pca, gaussian_pca, spaghetti_pca, sphere4_pca, sphere_pca, spiral_pca, train_or_load_pca, uniform_pca
import numpy as np
from sklearn.decomposition import PCA


class SampledDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e1_sampled_fmnist",
            sampling_step=1,
            N_train=50_000,
            N_val=5_000,
            N_test=5_000,
            copy_val_to_test=True,
            seed: int = 0,
            *args,
            **kwargs,
    ):
        self.copy_val_to_test = copy_val_to_test
        self.sampling_step = sampling_step
        dataset_name += f"_step{sampling_step}"

        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):

        dataset, labels = generate_sampled_dataset(
            self.sampling_step, self.n_train, type="FMNIST")

        self.n_train = dataset.shape[0] - self.n_val - self.n_test

        return {"dataset": dataset, "labels": labels}


class CrescentMoonPCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e7_crescent_moon",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            radius=3.00,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        dataset_name += f"_radius{radius}"
        self.radius = radius

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = crescent_moon_pca(
            self.n_train + self.n_val + self.n_test,
            self.pca,
            self.radius,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients
        }


class UniformPCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name: str = "e2_uniform_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=20,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = uniform_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}


class GaussianPCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e3_gaussian_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=20,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = gaussian_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}


class SpherePCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name: str = "e4_sphere_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=20,
            radius=1,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim
        self.radius = radius

        dataset_name += f"_radius{radius}"

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = sphere_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.radius,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}


class ExpPCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name: str = "e6_exp_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            seed=0,
            pca: Optional[PCA] = None,
    ):

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = exp_pca(
            self.n_train + self.n_val + self.n_test,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}


class SpiralPCADatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name: str = "e1_spiral_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            seed=0,
            pca: Optional[PCA] = None,
    ):

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = spiral_pca(
            self.n_train + self.n_val + self.n_test,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}

class StretchedDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name: str = "e5_stretched",
            N_train=50_000,
            N_val=5_000,
            N_test=5_000,
            copy_val_to_test=True,
            power=4,
            seed=0,
            *args,
            **kwargs,
    ):
        dataset_name += f"_power{power}"
        self.power = power
        self.copy_val_to_test = copy_val_to_test

        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):
        stretched_images, labels = generate_stretched_dataset(
            dataset_name='FMNIST', power=self.power)

        return {"dataset": stretched_images, "labels": labels}



class DownscaledDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e5_downscaled_fmnist",
            N_train=50_000,
            N_val=5_000,
            N_test=5_000,
            copy_val_to_test=True,
            seed: int = 0,
            *args,
            **kwargs,
    ):
        self.copy_val_to_test = copy_val_to_test
        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):
        downscaled_images, labels = generate_downscaled_dataset(
            dataset_name='FMNIST')

        return {"dataset": downscaled_images, "labels": labels}


class UpscaledDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e5_upscaled_fmnist",
            N_train=50_000,
            N_val=5_000,
            N_test=5_000,
            copy_val_to_test=True,
            seed: int = 0,
            *args,
            **kwargs,
    ):
        self.copy_val_to_test = copy_val_to_test
        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):
        downscaled_images, labels = generate_upscaled_dataset(
            dataset_name='FMNIST')

        return {"dataset": downscaled_images, "labels": labels}


class PaddedDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e5_padded_fmnist",
            N_train=50_000,
            N_val=5_000,
            N_test=5_000,
            copy_val_to_test=True,
            seed: int = 0,
            additional_dimension: int = 0,
            *args,
            **kwargs,
    ):
        dataset_name += f"_adddim{additional_dimension}"
        self.additional_dimension = additional_dimension
        self.copy_val_to_test = copy_val_to_test

        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):
        padded_images, labels = generate_padded_dataset(
            dataset_name="FMNIST", additional_dimensions=self.additional_dimension)

        return {"dataset": padded_images, "labels": labels}


class ArrowsDatasetGenerator(LIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e2_arrows",
            N_train=100_000,
            N_val=10_000,
            N_test=10_000,
            seed: int = 0,
            max_arrows: int = 4,
            size: int = 32,
            *args,
            **kwargs,
    ):
        self.max_arrows = max_arrows
        self.size = size

        super().__init__(dataset_root_dir, dataset_name,
                         N_train, N_val, N_test, seed, *args, **kwargs)

    def _generate_artifacts(self):
        dataset, lid = make_arrows_dataset(
            self.n_train + self.n_test + self.n_val,
            self.max_arrows,
            self.size,
            self.seed
        )

        return {"dataset": dataset, "lid": lid}


class Sphere4DatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e8_sphere4_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=6,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = sphere4_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}

class Gaussian4DatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e8_gaussian4_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=5,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = gaussian4_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}

class SpaghettiDatasetGenerator(PCALIDBenchmarkDatasetGenerator):

    def __init__(
            self,
            dataset_root_dir: str = "data",
            dataset_name="e8_spaghetti_pca",
            N_train=100_000,
            N_val=1_000,
            N_test=1_000,
            dim=20,
            seed=0,
            pca: Optional[PCA] = None,
    ):
        self.dim = dim

        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
            pca=pca,
        )

    def _generate_artifacts(self):
        dataset, lid, coefficients = spaghetti_pca(
            self.n_train + self.n_val + self.n_test,
            self.dim,
            self.pca,
            self.seed,
        )

        return {
            "dataset": dataset,
            "lid": lid,
            "coefficients": coefficients}



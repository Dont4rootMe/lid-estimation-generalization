from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from generators.utils.pca import train_or_load_pca
from sklearn.decomposition import PCA

class LIDBenchmarkDatasetGenerator(ABC):

    def __init__(
            self,
            dataset_root_dir: str,
            dataset_name: str,
            N_train: int,
            N_val: int,
            N_test: int,
            seed: int = 0,
            *args,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.seed = seed

        self.n_train = N_train
        self.n_val = N_val
        self.n_test = N_test
        self.copy_val_to_test = False

        self.dataset_dir = Path(dataset_root_dir) / dataset_name
        self.train_path = self.dataset_dir / "train"
        self.val_path = self.dataset_dir / "val"
        self.test_path = self.dataset_dir / "test"

    def generate(self):
        """Main method for dataset creation."""
        self._create_directories()
        artifacts_dict = self._generate_artifacts()
        self._save_artifacts(artifacts_dict)

    def _create_directories(self):
        subset_paths = [self.test_path, self.val_path, self.train_path]
        n_subsets = [self.n_train, self.n_val, self.n_test]

        for path_subset, n_subset in zip(subset_paths, n_subsets):
            if n_subset > 0:
                path_subset.mkdir(exist_ok=True, parents=True)

    def _generate_artifacts(self):
        self.logger.error(
            "You need to implement _generate_artifacts to use dataset generator.")
        raise NotImplementedError

    def _train_val_test_split(
            self,
            data: np.array
    ) -> tuple[np.array, np.array, np.array]:

        if data.shape[0] != self.n_train + self.n_val + self.n_test:
            self.logger.error(
                "Warning - some unused data left. Check your numbers/shapes.")
            raise Exception()
        x_train = data[:self.n_train, ...]
        x_val = data[self.n_train: self.n_train + self.n_val, ...]
        x_test = data[self.n_train + self.n_val: self.n_train +
                      self.n_val + self.n_test, ...]

        if self.copy_val_to_test:
            x_test = data[self.n_train: self.n_train + self.n_val, ...]

        return x_train, x_val, x_test

    def _save_artifacts(self, artifacts: dict[str, np.ndarray]):

        for file_name, data in artifacts.items():
            data_train, data_val, data_test = self._train_val_test_split(data)

            path_subsets = (self.train_path, self.val_path, self.test_path)
            x_subsets = (data_train, data_val, data_test)

            for path_subset, x_subset in zip(path_subsets, x_subsets):
                if x_subset.size > 0:
                    np.save(path_subset / f"{file_name}.npy", x_subset)


class PCALIDBenchmarkDatasetGenerator(LIDBenchmarkDatasetGenerator):
    def __init__(
            self,
            dataset_root_dir: str,
            dataset_name: str,
            N_train: int,
            N_val: int,
            N_test: int,
            seed: int = 0,
            pca: Optional[PCA]=None,
            *args,
            **kwargs
    ):
        if pca is None:
            pca_root_dir = f"{dataset_root_dir}/{dataset_name}"
            self.logger.info(f"Generating PCA data from scratch in {pca_root_dir}")
            self.pca = train_or_load_pca(
                seed=seed,
                pca_output_dir=pca_root_dir,
                pca_load_from_disk=(pca is not None),
            )
        else:
            self.pca = pca
        
        super().__init__(
            dataset_root_dir=dataset_root_dir,
            dataset_name=dataset_name,
            N_train=N_train,
            N_val=N_val,
            N_test=N_test,
            seed=seed,
        )

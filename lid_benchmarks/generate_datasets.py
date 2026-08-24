from functools import partial
from pathlib import Path
from generators.benchmarks import ArrowsDatasetGenerator, CrescentMoonPCADatasetGenerator, ExpPCADatasetGenerator, Gaussian4DatasetGenerator, GaussianPCADatasetGenerator, SampledDatasetGenerator, SpaghettiDatasetGenerator, Sphere4DatasetGenerator, SpherePCADatasetGenerator, SpiralPCADatasetGenerator, StretchedDatasetGenerator, UniformPCADatasetGenerator, DownscaledDatasetGenerator, PaddedDatasetGenerator, UpscaledDatasetGenerator

from tqdm.auto import tqdm

from generators.utils.pca import train_or_load_pca

DATASET_ROOT_DIR = "./data/benchmarks"


def prepare_all(dataset_root_dir: str):
    pca = train_or_load_pca(
        pca_output_dir=dataset_root_dir,
        pca_desired_class=7,
        pca_n_components=30,
        pca_load_from_disk=(Path(dataset_root_dir) / "pca.joblib").exists(),
    )

    generators_per_experiment = {
        "e1": [
            partial(SampledDatasetGenerator, sampling_step=s) for s in range(1, 14)
        ] + [partial(SpiralPCADatasetGenerator, pca=pca)],
        "e2": [
            partial(UniformPCADatasetGenerator, pca=pca),
            ArrowsDatasetGenerator,
        ],
        "e5": [
            *[partial(StretchedDatasetGenerator, power=power) for power in [4, 0.25]],
            DownscaledDatasetGenerator,
            UpscaledDatasetGenerator,
            *[partial(PaddedDatasetGenerator, additional_dimension=ad) for ad in [0, 4, 8]]
        ],
        "e6": [partial(ExpPCADatasetGenerator, pca=pca)],
        "e7": [partial(CrescentMoonPCADatasetGenerator, pca=pca)],
        "e8": [
            partial(Sphere4DatasetGenerator, pca=pca),
            partial(Gaussian4DatasetGenerator, pca=pca),
            partial(SpaghettiDatasetGenerator, pca=pca),
        ]
    }

    p_bar = tqdm(generators_per_experiment.items(), position=0, leave=True)
    for experiment_name, generators in p_bar:
        p_bar.set_description(
            f"Generating dataset for experiment {experiment_name}")
        p_bar_experiment = tqdm(generators, leave=False, position=1)
        for generator_class in p_bar_experiment:
            g = generator_class(dataset_root_dir=dataset_root_dir)
            p_bar_experiment.set_description(str(g.dataset_dir))
            g.generate()


if __name__ == "__main__":
    prepare_all(DATASET_ROOT_DIR)

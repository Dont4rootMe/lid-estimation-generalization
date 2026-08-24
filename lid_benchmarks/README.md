# [ICLR 2026] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation

This repository contains the code for generation of the datasets used in our LID estimation benchmarks, which are described in [Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](https://openreview.net/forum?id=ZEf03Uunvk) accepted to ICLR 2026.

## Downloading the benchmark

The benchmarks used in the paper (along with the used PCA output) can be found [here](https://drive.google.com/file/d/1mGzGUVa37AjUREHx_vFoPzl1OCLjPJ1Q/view?usp=share_link) (password: `LocalIntrinsicDimensionBenchmarks`)

| Folder | Section in [paper](https://openreview.net/forum?id=ZEf03Uunvk) | Experiment|
|--------|-----------|------------------|
| `e1_sampled_fmnist_step[1..13]` | 3.7| Esitmated LID vs sample size based on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)|
| `e1_spiral_pca` | 3.5 | Nearby manifolds: `Spiral (IDR)`|
| `e2_arrows` | 3.9 | Real-like dataset with known LID: `Arrows dataset (BMS)`|
| `e2_uniform_pca` | 3.3 | Boundaries of manifolds: `Uniform (IDR)`|
| `e5_padded_fmnist_adddim0` | 3.8 |Real-world dataset transformations: `FMNIST with added dimensions (ADI)` (+0d)|
| `e5_padded_fmnist_adddim4` | 3.8 |Real-world dataset transformations: `FMNIST with added dimensions (ADI)` (+4d)|
| `e5_padded_fmnist_adddim8` | 3.8 |Real-world dataset transformations: `FMNIST with added dimensions (ADI)` (+8d)|
| `e5_stretched_power0.25` | 3.8 | Real-world dataset transformations: `Stretched FMNIST dataset (ME)`|
| `e5_stretched_power4` | 3.8 | Real-world dataset transformations: `Stretched FMNIST dataset (ME)`|
| `e5_upscaled_fmnist` | 3.8 | Real-world dataset transformations: `Upscaled FMNIST (ASE)`|
| `e6_exp_pca` | 3.5 | Nearby manifolds: `Funnel (IDR)`|
| `e7_crescent_moon_radius3.0` | 3.4 | Thin manifolds: `Moon (IDR)`|
| `e8_gaussian4_pca` | 3.1 | Non-uniform densisties: `Gaussians (IDR)`|
| `e8_spaghetti_pca` | 3.2 | Manifold curvatures: `Spaghetti (IDR)`|
| `e8_sphere4_pca` | 3.2 | Manifold curvatures: `Spheres (IDR)` |


## Running the code to generate the benchmarks

> [!WARNING]  
> We don't guarantee the code will produce the same datasets as in the paper.
> If you want to use the exact datasets, please download them using the link above.

If you have [`uv`](https://docs.astral.sh/uv/) installed, simply run:
```{python}
uv run generate_datasets.py
```
This will:
- create project-scoped virtual environement,
- download base datasets (FMNIST),
- generate synthetic datasets.

All artifacts will appear in `data/`.
Inside, you will find the following structure:
- `base_datasets` contains the base datasets (e.g., FMNIST),
- `pca.joblib` is the output of PCA used in IDR experiments,
- `benchmarks` contains the benchmarks (separated to `train`, `val`, and `test`), which map to the experiments in the paper as follows.


## Citation
If you use our benchmarks in your research, please consider citing our paper:
```bibtex
@inproceedings{tempczyk2026why,
    title={Why We Need New Benchmarks for Local Intrinsic Dimension Estimation},
    author={Piotr Tempczyk and Dominik Filipiak and {\L}ukasz Garncarek and Ksawery Smoczy{\'n}ski and Adam Kurpisz},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=ZEf03Uunvk}
}
```

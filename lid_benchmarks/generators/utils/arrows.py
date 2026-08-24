from PIL import Image
import numpy as np
from tqdm import tqdm


def make_simple_arrow(r, g, b):
    x = np.zeros((9, 9, 3)).astype("uint8")
    x[3:8, 3:6] = [r, g, b]
    x[3, 1:8] = [r, g, b]
    x[2, 2:7] = [r, g, b]
    x[1, 3:6] = [r, g, b]
    return x


def rotate_and_translate(simple_arrow, a, tx, ty):
    simple_arrow = Image.fromarray(simple_arrow)
    rotated_image = simple_arrow.rotate(
        resample=Image.Resampling.BILINEAR, angle=a, translate=(tx, ty))
    return np.array(rotated_image)


def plot_arrow(x, i, j, a, r, g, b, img_size):
    int_i = int(i)
    int_j = int(j)
    simple_arrow = make_simple_arrow(r, g, b)
    rot_arrow = rotate_and_translate(simple_arrow, a, i - int_i, j - int_j)
    x[(int_i - 4):(int_i + 5), (int_j - 4):(int_j + 5)] = np.maximum(rot_arrow,
                                                                     x[(int_i - 4):(int_i + 5), (int_j - 4):(int_j + 5)])


def generate_image(img_size, n_arrows):
    x = np.zeros((img_size, img_size, 3)).astype('float')
    for _ in range(n_arrows):
        i, j = np.random.uniform(low=4, high=img_size-4, size=2)
        a = np.random.uniform(low=0, high=360, size=1)
        r, g, b = np.random.randint(100, 256, 3)
        plot_arrow(x, i, j, a, r, g, b, img_size)
    return x.clip(0, 255).astype("uint8")


def make_arrows_dataset(N, max_arrows, img_size, seed=0):
    np.random.seed(seed)

    dataset = []
    lid = []
    n_arrows_all = np.random.randint(low=1, high=max_arrows + 1, size=N).tolist()

    for n_arrows in tqdm(n_arrows_all):       
        dataset.append(generate_image(img_size, n_arrows))
        lid.append(n_arrows * 6)

    data = np.stack(dataset, axis=0)
    return data, np.array(lid)

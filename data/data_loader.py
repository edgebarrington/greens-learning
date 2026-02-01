import numpy as np

class PoissonDataset:
    """
    Lightweight dataset wrapper for Poisson (f, u) pairs.
    Read-only by design.
    """

    def __init__(self, path):
        data = np.load(path, allow_pickle=True)

        self.x = data["x"]
        self.f = data["f"]
        self.u = data["u"]
        self.coeffs = data["coeffs"]
        self.meta = data["meta"].item()

        self.num_samples = self.f.shape[0]
        self.N = self.f.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "x": self.x,
            "f": self.f[idx],
            "u": self.u[idx],
            "coeffs": self.coeffs[idx],
        }


class DataLoader:
    """
    Simple NumPy data loader with batching and optional shuffling.
    """

    def __init__(self, dataset, batch_size=32, shuffle=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.indices = np.arange(len(dataset))
        self._reset()

    def _reset(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices)
        self.ptr = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration

        batch_idx = self.indices[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size

        batch = [self.dataset[i] for i in batch_idx]

        return {
            "x": batch[0]["x"],
            "f": np.stack([b["f"] for b in batch]),
            "u": np.stack([b["u"] for b in batch]),
            "coeffs": np.stack([b["coeffs"] for b in batch]),
        }

import os
import sys
sys.path.insert(0, '/eos/home-m/majorgen/pquant_dev/src')
import torch
from torch.utils.data import DataLoader, IterableDataset

class DummyDataset(IterableDataset):
    def __iter__(self):
        for i in range(100):
            yield torch.zeros(10)

def worker_init_fn(worker_id):
    backend = os.environ.get('KERAS_BACKEND', 'NOT SET')
    print(f"Worker {worker_id}: KERAS_BACKEND={backend}", flush=True)
    if backend == 'NOT SET':
        raise RuntimeError(f"Worker {worker_id}: KERAS_BACKEND not set!")
    try:
        import pquant
        print(f"Worker {worker_id}: pquant imported successfully", flush=True)
    except Exception as e:
        print(f"Worker {worker_id}: pquant import FAILED: {e}", flush=True)
        raise

if __name__ == '__main__':
    os.environ['KERAS_BACKEND'] = 'torch'
    print(f"Main process: KERAS_BACKEND={os.environ.get('KERAS_BACKEND')}")
    dataset = DummyDataset()
    loader = DataLoader(dataset, num_workers=2, worker_init_fn=worker_init_fn)
    try:
        batch = next(iter(loader))
        print("Done — no error")
    except Exception as e:
        print(f"Error: {e}")

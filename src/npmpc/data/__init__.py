import json
import os
import torch


class DatasetBase(torch.utils.data.Dataset):
    """Disk-backed dataset of pre-transformed .pt files; scaling read from metadata.json.
    The path supports slice syntax, e.g. 'datasets/.../50:200' selects elements 50-199."""

    def __init__(self, path: str):
        self.data_dir, self._start, self._end = self._parse_path(path)
        with open(os.path.join(self.data_dir, '_metadata.json')) as f:
            self.metadata = json.load(f)

        total = self.metadata['num_systems']
        if self._end is None:
            self._end = total
        self._end = min(self._end, total)
        self._start = min(self._start, self._end)

        scaling = self.metadata['scaling']
        self.x_mean = torch.tensor(scaling['x']['mean'])
        self.x_scale = torch.tensor(scaling['x']['std'])
        self.y_mean = torch.tensor(scaling['y']['mean'])
        self.y_scale = torch.tensor(scaling['y']['std'])

    @staticmethod
    def _parse_path(path: str):
        """Parse 'dir/start:end' into (dir, start, end). Returns (path, 0, None) if no slice."""
        path = path.rstrip('/')
        base = os.path.basename(path)
        if ':' in base:
            data_dir = os.path.dirname(path)
            parts = base.split(':')
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else None
            return data_dir, start, end
        return path, 0, None

    def __len__(self):
        return self._end - self._start

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_dir, f'{self._start + idx:06d}.pt'),
                          weights_only=True)

    def get_scaling(self):
        return {
            'x': {'mean': self.x_mean, 'scale': self.x_scale},
            'y': {'mean': self.y_mean, 'scale': self.y_scale},
        }

from npmpc.data.furuta import FurutaDataset

DATASET_REGISTRY = {
    'furuta': FurutaDataset,
}

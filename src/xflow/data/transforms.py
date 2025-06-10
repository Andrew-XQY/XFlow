# src/xflow/data/transforms.py

from __future__ import annotations
import random
import itertools
from .loader import BasePipeline
class ShufflePipeline(BasePipeline):
    def __init__(self, base: BasePipeline, buffer_size: int):
        self.base = base
        self.buf = buffer_size

    def __iter__(self):
        it = self.base.__iter__()
        buf = list(itertools.islice(it, self.buf))
        random.shuffle(buf)
        for x in buf:
            yield x
        for x in it:
            buf[random.randrange(self.buf)] = x
            random.shuffle(buf)
            yield buf.pop()

    def __len__(self):
        return len(self.base)

    def to_framework_dataset(self):
        return self.base.to_framework_dataset().shuffle(self.buf)
class BatchPipeline(BasePipeline):
    def __init__(self, base: BasePipeline, batch_size: int):
        self.base = base
        self.bs = batch_size

    def __iter__(self):
        it = self.base.__iter__()
        while True:
            batch = list(itertools.islice(it, self.bs))
            if not batch:
                break
            yield batch

    def __len__(self):
        return (len(self.base) + self.bs - 1) // self.bs

    def to_framework_dataset(self):
        return self.base.to_framework_dataset().batch(self.bs)
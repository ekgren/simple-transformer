# Copyright (c) 2022, NORTHALGO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Most of the code here has been copied from:
#   fairseq/fairseq/data/indexed_dataset.py

# with some modifications:

# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import os
import logging
import shutil
import struct
from functools import lru_cache
from itertools import accumulate, islice
from dataclasses import dataclass

import numpy as np
import torch


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'

def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                logging.info("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            logging.info("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            logging.info("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            logging.info("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    # def __setstate__(self, state):
    #     self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            logging.info("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        logging.info("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        logging.info("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.
        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

class IterableBinDataset(torch.utils.data.IterableDataset):
    def __init__(self, mmap_ds, max_len, subset=None, repeat=False):
        super().__init__()
        self.ds = mmap_ds
        if subset is None:
            self.itoi = np.arange(len(self.ds), dtype=np.uint64)
        else:
            self.itoi = np.array(subset, copy=True)
        
        np.random.shuffle(self.itoi)
        self.max_len = max_len
        self.repeat = repeat

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Multiworker not implemented (and not worth it?)"

        ix = 0 
        
        tok_buff = None
        offset = None
        
        written = None
        vacant = None
        doc_id = None

        doc_ids = np.empty(self.max_len, dtype=np.int64)
        tok_ids = np.empty(self.max_len, dtype=np.int64)

        def populate_token_buffer():
            nonlocal tok_buff, offset
            tok_buff = self.ds.get(self.itoi[ix])
            offset = 0

        def reset_batch():
            nonlocal vacant, written, doc_id
            written = 0
            vacant = self.max_len
            doc_id = 1
        
        populate_token_buffer()
        reset_batch()
        
        while True:
            size = len(tok_buff) - offset
            read = min(size, vacant)

            tok_ids[written:written+read] = tok_buff[offset:offset+read]
            doc_ids[written:written+read].fill(doc_id)
            
            written += read
            offset += read
            vacant -= read

            # assert written + vacant = max_len, "This always holds"
            
            # If we read the whole document
            # Increment document ix and read in a new document
            if size == read:
                
                # If we are repeating: 
                # Shuffle seen document indices on the fly
                # (Fisher-Yates)
                if self.repeat:
                    jx = np.random.randint(ix+1)
                    self.itoi[[ix, jx]] = self.itoi[[jx, ix]]

                ix += 1
                # When done with epoch
                #  If set to repeat: Shuffle and Repeat
                #  Else: Break
                if ix >= len(self.itoi):
                    if self.repeat:
                        ix = 0 
                    else:
                        break
                doc_id += 1
                populate_token_buffer()
            
            # If we filled up the batch
            # concatenate parts, yield the result
            # Reset tracking variables
            if vacant == 0:
                yield np.array(tok_ids, copy=True), np.array(doc_ids, copy=True)
                reset_batch()

if __name__ == '__main__':
    import sentencepiece as spm
    import tqdm
    sp = spm.SentencePieceProcessor(model_file='/home/amaru/data/64k_new.model')
    mmap_ds = MMapIndexedDataset('/home/amaru/data/articles_nordic.jsonl_text_document')
    ds = IterableBinDataset(mmap_ds, 16384, repeat=True)
    dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=None)
    for ti, di in tqdm.tqdm(dl):
        pass

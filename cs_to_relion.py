#!/usr/bin/env python3
"""
Convert a cryoSPARC exported .cs particle file into a Relion .star and .mrcs stack.

The script attempts to load the .cs using numpy (cryoSPARC .cs files are NumPy arrays),
finds the fields containing the particle blob path and index, extracts the images
from the referenced mrc/mrcs files, writes an output .mrcs stack and a minimal
Relion star file containing `rlnImageName` entries pointing to the new stack.

If you have `pyem` installed and prefer to use its conversion tools, install it
with `pip install pyem` and consider using its `csparc2star` utilities.
"""

import argparse
import os
import sys
import numpy as np
import mrcfile
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import multiprocessing as mp


def process_group(task):
    """Worker: write assigned slices from a single candidate file into the shared memmap."""
    candidate, entries, temp_npy, total, ny, nx, dtype_str = task
    dtype = np.dtype(dtype_str)
    mm = np.memmap(temp_npy, dtype=dtype, mode='r+', shape=(total, ny, nx))
    count = 0
    with mrcfile.open(candidate, permissive=True) as m:
        data = m.data
        for out_idx, img_idx in entries:
            if data.ndim == 3:
                if img_idx >= data.shape[0]:
                    raise IndexError(f"Index {img_idx} out of bounds for file {candidate}")
                mm[out_idx] = data[img_idx]
            elif data.ndim == 2:
                if img_idx != 0:
                    raise IndexError(f"Index {img_idx} invalid for single-image file {candidate}")
                mm[out_idx] = data
            else:
                raise ValueError(f"Unsupported MRC data dimensions: {data.ndim} in {candidate}")
            count += 1
    mm.flush()
    return count


def find_field(names, keywords):
    if names is None:
        return None
    for k in keywords:
        for n in names:
            if k in n:
                return n
    return None


def decode_field(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)


def normalize_blob_path(p):
    """Normalize blob-style paths commonly found in cryoSPARC exports.

    Strips leading characters like '>' or '|' and prefixes like 'blob:' or 'blobstore:'.
    Preserves absolute paths that start with '/'.
    """
    if p is None:
        return p
    s = p.strip()
    had_leading_slash = s.startswith('/')
    # remove common leading junk characters
    s = s.lstrip('>|')
    # remove common blob prefixes
    for prefix in ('blobstore:', 'blob:'):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # if it originally was absolute, restore leading slash
    if had_leading_slash and not s.startswith('/'):
        s = '/' + s.lstrip('/')
    return s


def load_cs_array(cs_path):
    # np.load handles both .npy and .npz; allow_pickle to be tolerant
    arr = np.load(cs_path, allow_pickle=True)
    # If .npz, pick the first array
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = arr.files
        if not keys:
            raise ValueError("Empty .npz file")
        arr = arr[keys[0]]
    return arr


def write_star(output_star, image_basename, n_particles):
    with open(output_star, 'w') as f:
        f.write('data_\n')
        f.write('loop_\n')
        f.write('_rlnImageName\n')
        for i in range(n_particles):
            f.write(f"{i:06d}@{image_basename}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert cryoSPARC .cs to Relion .star + .mrcs')
    parser.add_argument('--cs', required=True, help='Input cryoSPARC exported .cs file')
    parser.add_argument('--input_dir', default=None, help='Base directory to resolve blob paths (optional)')
    parser.add_argument('--output_mrcs', required=True, help='Output mrcs stack file path')
    parser.add_argument('--output_star', required=True, help='Output Relion star file path')
    parser.add_argument('--path_field', default=None, help='(Optional) name of field containing blob path')
    parser.add_argument('--idx_field', default=None, help='(Optional) name of field containing image index')

    args = parser.parse_args()

    cs_path = args.cs
    if not os.path.exists(cs_path):
        print(f"ERROR: .cs file not found: {cs_path}")
        sys.exit(2)

    arr = load_cs_array(cs_path)

    # Determine dtype field names
    names = None
    if hasattr(arr, 'dtype') and arr.dtype.names is not None:
        names = list(arr.dtype.names)

    # Heuristics for fields
    path_field = args.path_field or find_field(names, ['blob/path', 'blob_path', 'path'])
    idx_field = args.idx_field or find_field(names, ['blob/idx', 'blob_idx', 'idx', 'index'])

    if path_field is None or idx_field is None:
        # Try object-array of dicts
        sample = arr[0]
        if isinstance(sample, (bytes, str)):
            print('ERROR: .cs appears to contain only paths; this script expects structured particle entries.')
            sys.exit(3)
        if isinstance(sample, dict):
            # keys of dict
            keys = list(sample.keys())
            path_field = path_field or find_field(keys, ['blob/path', 'blob_path', 'path'])
            idx_field = idx_field or find_field(keys, ['blob/idx', 'blob_idx', 'idx', 'index'])

    if path_field is None or idx_field is None:
        print('ERROR: Could not determine path or idx fields from .cs. Found fields:', names)
        sys.exit(4)

    # Prepare references list (candidate path, image index) for all particles
    refs = []
    base_dir = args.input_dir or os.path.dirname(os.path.abspath(cs_path))
    for i, entry in enumerate(arr):
        if isinstance(entry, np.void):
            raw_path = entry[path_field]
            raw_idx = entry[idx_field]
        elif isinstance(entry, dict):
            raw_path = entry[path_field]
            raw_idx = entry[idx_field]
        else:
            try:
                raw_path = entry[names.index(path_field)]
                raw_idx = entry[names.index(idx_field)]
            except Exception:
                raise ValueError('Unexpected entry type in .cs array; cannot extract fields')

        path_str = decode_field(raw_path)
        path_str = normalize_blob_path(path_str)
        if not os.path.isabs(path_str):
            candidate = os.path.join(base_dir, path_str)
        else:
            candidate = path_str

        if not os.path.exists(candidate):
            candidate_alt = os.path.join(base_dir, os.path.basename(path_str))
            if os.path.exists(candidate_alt):
                candidate = candidate_alt
            else:
                raise FileNotFoundError(f"Referenced MRC/MRCS not found: {candidate}")

        img_idx = int(raw_idx)
        refs.append((candidate, img_idx))

    total = len(refs)

    # Determine image shape from first referenced image
    sample_candidate, sample_idx = refs[0]
    with mrcfile.open(sample_candidate, permissive=True) as m:
        sample_data = m.data
        if sample_data.ndim == 3:
            ny, nx = sample_data.shape[1], sample_data.shape[2]
        elif sample_data.ndim == 2:
            ny, nx = sample_data.shape[0], sample_data.shape[1]
        else:
            raise ValueError(f"Unsupported MRC data dimensions: {sample_data.ndim} in {sample_candidate}")

    # Pre-allocate memmap-backed array to avoid holding everything in RAM
    out_mrcs = args.output_mrcs
    os.makedirs(os.path.dirname(os.path.abspath(out_mrcs)) or '.', exist_ok=True)
    temp_npy = out_mrcs + '.npy.tmp'
    # Use float32 memmap to avoid creating a full in-memory float32 copy later.
    # Assigning slices will cast to float32 per-slice (small, chunked copies).
    src_dtype = sample_data.dtype
    out_dtype = np.float32
    mm = np.memmap(temp_npy, dtype=out_dtype, mode='w+', shape=(total, ny, nx))

    # Group references by candidate file to open each file once
    from collections import defaultdict
    groups = defaultdict(list)
    for out_idx, (candidate, img_idx) in enumerate(refs):
        groups[candidate].append((out_idx, img_idx))

    # Prepare multiprocessing (per-file) tasks
    parser_workers = 1
    try:
        parser_workers = int(os.environ.get('CS_TO_RELION_WORKERS', '1'))
    except Exception:
        parser_workers = 1
    # allow override from CLI
    workers = getattr(args, 'workers', None)
    if workers is None:
        workers = parser_workers

    tasks = []
    for candidate, entries in groups.items():
        tasks.append((candidate, entries, temp_npy, total, ny, nx, str(out_dtype)))

    processed_particles = 0
    if workers and int(workers) > 1:
        pool_workers = int(workers)
        with mp.Pool(pool_workers) as pool:
            if tqdm is not None:
                for cnt in tqdm(pool.imap_unordered(process_group, tasks), total=len(tasks), desc='Files', unit='files'):
                    processed_particles += int(cnt)
            else:
                for cnt in pool.imap_unordered(process_group, tasks):
                    processed_particles += int(cnt)
    else:
        # single-process fallback
        if tqdm is not None:
            task_iter = tqdm(tasks, desc='Files', unit='files')
        else:
            task_iter = tasks
        for task in task_iter:
            processed_particles += process_group(task)

    if processed_particles != total:
        print(f"Warning: processed {processed_particles} particles but expected {total}")

    # Flush memmap to disk
    mm.flush()

    # Write final mrcs using memory-mapped MRC to avoid loading full array into RAM.
    # mrc_mode=2 corresponds to float32.
    chunk_size = 1000
    mrc = mrcfile.new_mmap(out_mrcs, shape=(total, ny, nx), mrc_mode=2, overwrite=True)

    # Copy data in chunks from temp memmap into the output MRC memmap
    print(f"Writing {total} particles to {out_mrcs} ...")
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        mrc.data[start:end] = mm[start:end]

    # Compute header stats in chunks to avoid OOM
    running_sum = np.float64(0)
    running_sq_sum = np.float64(0)
    running_min = np.float32(np.inf)
    running_max = np.float32(-np.inf)
    n_elements = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = mrc.data[start:end]
        running_sum += np.float64(chunk.sum())
        running_sq_sum += np.float64((chunk.astype(np.float64) ** 2).sum())
        cmin, cmax = chunk.min(), chunk.max()
        if cmin < running_min:
            running_min = cmin
        if cmax > running_max:
            running_max = cmax
        n_elements += chunk.size

    mean_val = running_sum / n_elements
    rms_val = np.sqrt(max(running_sq_sum / n_elements - mean_val ** 2, 0))
    mrc.header.dmin = np.float32(running_min)
    mrc.header.dmax = np.float32(running_max)
    mrc.header.dmean = np.float32(mean_val)
    mrc.header.rms = np.float32(rms_val)
    mrc.flush()
    mrc.close()

    # Remove temp npy
    try:
        os.remove(temp_npy)
    except Exception:
        pass

    # Write minimal star file
    out_star = args.output_star
    os.makedirs(os.path.dirname(os.path.abspath(out_star)) or '.', exist_ok=True)
    write_star(out_star, os.path.basename(out_mrcs), total)

    print(f"Wrote {total} particles to {out_mrcs} and star {out_star}")


if __name__ == '__main__':
    main()

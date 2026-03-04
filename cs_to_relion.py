#!/usr/bin/env python3
"""
Convert a cryoSPARC exported .cs particle file into a Relion .star and .mrcs stack.

Reads a cryoSPARC .cs (NumPy structured array), resolves blob paths, extracts
particle images from referenced MRC/MRCS files, writes a combined .mrcs stack and
a minimal Relion .star file.

Performance features:
  - Auto-detects CPU count and available RAM to pick sensible defaults.
  - Groups reads by source file (each file opened once).
  - Sorts indices within each file for sequential I/O.
  - Batches contiguous index ranges into single numpy slice reads.
  - Writes directly to a memory-mapped output MRC (no temp file, no full-array copy).
  - Optional multiprocessing (--workers) for parallel per-file reads.
  - Computes MRC header stats in chunks to avoid OOM.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
import numpy as np
import mrcfile

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import multiprocessing as mp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_available_memory_gb():
    """Return available system memory in GiB (Linux). Falls back to 4 GiB."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GiB
    except Exception:
        pass
    return 4.0


def auto_workers(num_files):
    """Pick a sensible default worker count based on CPUs and file count."""
    cpus = os.cpu_count() or 1
    # Use at most half the CPUs (I/O bound; diminishing returns beyond that)
    # and never more workers than files.
    return max(1, min(cpus // 2, num_files))


def auto_chunk_size(ny, nx, dtype=np.float32):
    """Pick chunk size (number of images) that fits comfortably in ~25% of free RAM."""
    avail = get_available_memory_gb() * 1024**3  # bytes
    image_bytes = ny * nx * np.dtype(dtype).itemsize
    target = avail * 0.25
    chunk = max(100, int(target / image_bytes))
    return chunk


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
    """Normalize blob-style paths from cryoSPARC exports.

    Strips leading '>' / '|' and prefixes like 'blob:' or 'blobstore:'.
    Preserves absolute paths.
    """
    if p is None:
        return p
    s = p.strip()
    had_leading_slash = s.startswith('/')
    s = s.lstrip('>|')
    for prefix in ('blobstore:', 'blob:'):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if had_leading_slash and not s.startswith('/'):
        s = '/' + s.lstrip('/')
    return s


def load_cs_array(cs_path):
    arr = np.load(cs_path, allow_pickle=True)
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


# ---------------------------------------------------------------------------
# Worker (reads one source file, writes disjoint slices into output memmap)
# ---------------------------------------------------------------------------

def _batch_contiguous(entries):
    """Given sorted (out_idx, img_idx) pairs, yield contiguous batches.

    A batch is contiguous when both out_idx and img_idx increment by 1 for
    every successive entry.  Contiguous batches can be copied with a single
    numpy slice assignment instead of per-image loops.
    """
    if not entries:
        return
    batch_start_out = entries[0][0]
    batch_start_img = entries[0][1]
    batch_len = 1
    for i in range(1, len(entries)):
        prev_out, prev_img = entries[i - 1]
        cur_out, cur_img = entries[i]
        if cur_out == prev_out + 1 and cur_img == prev_img + 1:
            batch_len += 1
        else:
            yield batch_start_out, batch_start_img, batch_len
            batch_start_out = cur_out
            batch_start_img = cur_img
            batch_len = 1
    yield batch_start_out, batch_start_img, batch_len


def process_group(task):
    """Worker: open one MRC file, copy its required slices into the output MRC memmap."""
    candidate, entries, out_mrcs_path, total, ny, nx = task

    # Open the output MRC as a raw memmap (skip the 1024-byte MRC header)
    mm = np.memmap(out_mrcs_path, dtype=np.float32, mode='r+',
                   offset=1024, shape=(total, ny, nx))

    # Sort entries by img_idx for sequential reads from source file
    entries_sorted = sorted(entries, key=lambda x: x[1])

    count = 0
    with mrcfile.open(candidate, permissive=True) as m:
        data = m.data
        if data.ndim == 3:
            for out_start, img_start, length in _batch_contiguous(entries_sorted):
                if img_start + length > data.shape[0]:
                    raise IndexError(
                        f"Index range {img_start}:{img_start+length} out of bounds "
                        f"for file {candidate} with {data.shape[0]} images")
                mm[out_start:out_start + length] = data[img_start:img_start + length]
                count += length
        elif data.ndim == 2:
            for out_idx, img_idx in entries_sorted:
                if img_idx != 0:
                    raise IndexError(
                        f"Index {img_idx} invalid for single-image file {candidate}")
                mm[out_idx] = data
                count += 1
        else:
            raise ValueError(
                f"Unsupported MRC data dimensions: {data.ndim} in {candidate}")

    mm.flush()
    del mm
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert cryoSPARC .cs to Relion .star + .mrcs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cs', required=True,
                        help='Input cryoSPARC exported .cs file')
    parser.add_argument('--input_dir', default=None,
                        help='Base directory to resolve blob paths (default: directory of .cs)')
    parser.add_argument('--output_mrcs', required=True,
                        help='Output mrcs stack file path')
    parser.add_argument('--output_star', required=True,
                        help='Output Relion star file path')
    parser.add_argument('--path_field', default=None,
                        help='Name of field containing blob path')
    parser.add_argument('--idx_field', default=None,
                        help='Name of field containing image index')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto = CPUs/2, capped by file count)')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for final copy/stats (default: auto from available RAM)')

    args = parser.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load .cs
    # ------------------------------------------------------------------
    cs_path = args.cs
    if not os.path.exists(cs_path):
        print(f"ERROR: .cs file not found: {cs_path}")
        sys.exit(2)

    arr = load_cs_array(cs_path)

    names = None
    if hasattr(arr, 'dtype') and arr.dtype.names is not None:
        names = list(arr.dtype.names)

    path_field = args.path_field or find_field(names, ['blob/path', 'blob_path', 'path'])
    idx_field = args.idx_field or find_field(names, ['blob/idx', 'blob_idx', 'idx', 'index'])

    if path_field is None or idx_field is None:
        sample = arr[0]
        if isinstance(sample, (bytes, str)):
            print('ERROR: .cs contains only paths; expected structured particle entries.')
            sys.exit(3)
        if isinstance(sample, dict):
            ks = list(sample.keys())
            path_field = path_field or find_field(ks, ['blob/path', 'blob_path', 'path'])
            idx_field = idx_field or find_field(ks, ['blob/idx', 'blob_idx', 'idx', 'index'])

    if path_field is None or idx_field is None:
        print('ERROR: Could not determine path/idx fields. Fields found:', names)
        sys.exit(4)

    # ------------------------------------------------------------------
    # Build reference list
    # ------------------------------------------------------------------
    refs = []
    base_dir = args.input_dir or os.path.dirname(os.path.abspath(cs_path))
    for entry in arr:
        if isinstance(entry, np.void):
            raw_path, raw_idx = entry[path_field], entry[idx_field]
        elif isinstance(entry, dict):
            raw_path, raw_idx = entry[path_field], entry[idx_field]
        else:
            try:
                raw_path = entry[names.index(path_field)]
                raw_idx = entry[names.index(idx_field)]
            except Exception:
                raise ValueError('Unexpected entry type in .cs array')

        path_str = normalize_blob_path(decode_field(raw_path))
        candidate = (os.path.join(base_dir, path_str)
                     if not os.path.isabs(path_str) else path_str)

        if not os.path.exists(candidate):
            alt = os.path.join(base_dir, os.path.basename(path_str))
            if os.path.exists(alt):
                candidate = alt
            else:
                raise FileNotFoundError(f"MRC not found: {candidate}")

        refs.append((candidate, int(raw_idx)))

    total = len(refs)

    # ------------------------------------------------------------------
    # Probe image dimensions
    # ------------------------------------------------------------------
    sample_candidate = refs[0][0]
    with mrcfile.open(sample_candidate, permissive=True) as m:
        sd = m.data
        if sd.ndim == 3:
            ny, nx = sd.shape[1], sd.shape[2]
        elif sd.ndim == 2:
            ny, nx = sd.shape
        else:
            raise ValueError(f"Unsupported ndim={sd.ndim} in {sample_candidate}")

    # ------------------------------------------------------------------
    # Group by source file; sort indices inside each group
    # ------------------------------------------------------------------
    groups = defaultdict(list)
    for out_idx, (candidate, img_idx) in enumerate(refs):
        groups[candidate].append((out_idx, img_idx))

    num_files = len(groups)

    # ------------------------------------------------------------------
    # Auto-detect resources
    # ------------------------------------------------------------------
    avail_mem = get_available_memory_gb()
    workers = args.workers if args.workers is not None else auto_workers(num_files)
    chunk_size = args.chunk_size if args.chunk_size is not None else auto_chunk_size(ny, nx)

    print(f"Particles: {total}  |  Source files: {num_files}  |  "
          f"Image size: {ny}x{nx}")
    print(f"Available RAM: {avail_mem:.1f} GiB  |  Workers: {workers}  |  "
          f"Chunk size: {chunk_size}")

    # ------------------------------------------------------------------
    # Create output MRC memmap (no temp file needed)
    # ------------------------------------------------------------------
    out_mrcs = args.output_mrcs
    os.makedirs(os.path.dirname(os.path.abspath(out_mrcs)) or '.', exist_ok=True)

    # mrc_mode=2 → float32
    mrc = mrcfile.new_mmap(out_mrcs, shape=(total, ny, nx), mrc_mode=2,
                           overwrite=True)
    mrc.flush()  # ensure file is sized on disk before workers write to it

    # ------------------------------------------------------------------
    # Build tasks
    # ------------------------------------------------------------------
    tasks = [(cand, entries, os.path.abspath(out_mrcs), total, ny, nx)
             for cand, entries in groups.items()]

    # ------------------------------------------------------------------
    # Process (parallel or single)
    # ------------------------------------------------------------------
    processed = 0
    if workers > 1:
        print(f"Extracting particles with {workers} parallel workers ...")
        with mp.Pool(workers) as pool:
            if tqdm is not None:
                for cnt in tqdm(pool.imap_unordered(process_group, tasks),
                                total=num_files, desc='Files', unit='files'):
                    processed += cnt
            else:
                for cnt in pool.imap_unordered(process_group, tasks):
                    processed += cnt
                    if processed % 5000 < cnt:
                        print(f"  {processed}/{total} particles ...")
    else:
        print("Extracting particles (single process) ...")
        it = tqdm(tasks, desc='Files', unit='files') if tqdm else tasks
        for task in it:
            processed += process_group(task)

    if processed != total:
        print(f"Warning: processed {processed} particles, expected {total}")

    # ------------------------------------------------------------------
    # Recompute header stats in chunks (avoids OOM)
    # ------------------------------------------------------------------
    print("Computing MRC header statistics ...")
    running_sum = np.float64(0)
    running_sq  = np.float64(0)
    running_min = np.float32(np.inf)
    running_max = np.float32(-np.inf)
    n_elem = 0

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = mrc.data[start:end]
        running_sum += np.float64(chunk.sum())
        running_sq  += np.float64((chunk.astype(np.float64) ** 2).sum())
        cmin, cmax = chunk.min(), chunk.max()
        if cmin < running_min:
            running_min = cmin
        if cmax > running_max:
            running_max = cmax
        n_elem += chunk.size

    mean_val = running_sum / n_elem
    rms_val  = np.sqrt(max(running_sq / n_elem - mean_val ** 2, 0))
    mrc.header.dmin  = np.float32(running_min)
    mrc.header.dmax  = np.float32(running_max)
    mrc.header.dmean = np.float32(mean_val)
    mrc.header.rms   = np.float32(rms_val)
    mrc.flush()
    mrc.close()

    # ------------------------------------------------------------------
    # Star file
    # ------------------------------------------------------------------
    out_star = args.output_star
    os.makedirs(os.path.dirname(os.path.abspath(out_star)) or '.', exist_ok=True)
    write_star(out_star, os.path.basename(out_mrcs), total)

    elapsed = time.time() - t0
    print(f"Wrote {total} particles to {out_mrcs} and {out_star} "
          f"in {elapsed:.1f}s ({elapsed/total:.3f} s/particle)")


if __name__ == '__main__':
    main()

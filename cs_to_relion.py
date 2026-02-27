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

    # Prepare images list
    images = []

    # Base dir resolution
    base_dir = args.input_dir or os.path.dirname(os.path.abspath(cs_path))

    total = len(arr)
    if tqdm is not None:
        index_iter = range(total)
        index_iter = tqdm(index_iter, desc='Extracting', unit='particles')
    else:
        index_iter = range(total)

    for i in index_iter:
        entry = arr[i]
        # Depending on type, entry may be structured numpy scalar or dict
        if isinstance(entry, np.void):
            raw_path = entry[path_field]
            raw_idx = entry[idx_field]
        elif isinstance(entry, dict):
            raw_path = entry[path_field]
            raw_idx = entry[idx_field]
        else:
            # fallback: could be tuple-like
            try:
                raw_path = entry[names.index(path_field)]
                raw_idx = entry[names.index(idx_field)]
            except Exception:
                raise ValueError('Unexpected entry type in .cs array; cannot extract fields')

        path_str = decode_field(raw_path)
        path_str = normalize_blob_path(path_str)
        # Resolve relative or blob-style paths
        if not os.path.isabs(path_str):
            candidate = os.path.join(base_dir, path_str)
        else:
            candidate = path_str

        if not os.path.exists(candidate):
            # try to strip leading slashes or blob prefixes
            candidate_alt = os.path.join(base_dir, os.path.basename(path_str))
            if os.path.exists(candidate_alt):
                candidate = candidate_alt
            else:
                raise FileNotFoundError(f"Referenced MRC/MRCS not found: {candidate}")

        img_idx = int(raw_idx)

        with mrcfile.open(candidate, permissive=True) as mrc:
            data = mrc.data
            if data.ndim == 3:
                if img_idx >= data.shape[0]:
                    raise IndexError(f"Index {img_idx} out of bounds for file {candidate}")
                images.append(data[img_idx].copy())
            elif data.ndim == 2:
                if img_idx != 0:
                    raise IndexError(f"Index {img_idx} invalid for single-image file {candidate}")
                images.append(data.copy())
            else:
                raise ValueError(f"Unsupported MRC data dimensions: {data.ndim} in {candidate}")

    # Stack and write mrcs
    stack = np.stack(images, axis=0)
    out_mrcs = args.output_mrcs
    os.makedirs(os.path.dirname(os.path.abspath(out_mrcs)) or '.', exist_ok=True)
    with mrcfile.new(out_mrcs, overwrite=True) as m:
        m.set_data(stack.astype(np.float32))

    # Write minimal star file
    out_star = args.output_star
    os.makedirs(os.path.dirname(os.path.abspath(out_star)) or '.', exist_ok=True)
    write_star(out_star, os.path.basename(out_mrcs), len(images))

    print(f"Wrote {len(images)} particles to {out_mrcs} and star {out_star}")


if __name__ == '__main__':
    main()

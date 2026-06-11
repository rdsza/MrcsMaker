import os
import argparse
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm

# Number of slices read/written per vectorised numpy operation.
# Larger values reduce Python-loop overhead; smaller values limit peak RAM.
_COPY_CHUNK = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def parse_image_reference(ref_string):
    """
    Parse Relion's image reference format: 000index@filename.mrc
    Returns (index, filename)
    """
    if match := re.match(r'(\d+)@(.+)', ref_string):
        return int(match[1]) - 1, match[2]  # Relion uses 1-based indices
    else:
        raise ValueError(f"Invalid image reference format: {ref_string}")


def read_star_file(star_file_path, image_column_name):
    """
    Read a Relion star file and extract the image references.
    Returns a DataFrame with the star file data.
    """
    # Read the star file
    with open(star_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the data block
    data_start = None
    header_start = None
    column_names = []
    
    for i, line in enumerate(lines):
        if line.strip() == "data_particles" or line.strip() == "data_":
            data_start = i
        elif line.strip() == "loop_" and data_start is not None:
            header_start = i + 1
        elif header_start is not None and line.strip() and line.strip()[0] == '_':
            column_names.append(line.strip().split()[0][1:])  # Remove leading '_'
        elif header_start is not None and column_names and (not line.strip() or line.strip()[0] != '_'):
            data_start = i
            break
    
    if header_start is None or data_start is None:
        raise ValueError("Could not parse star file format")
    
    # Read the data rows
    data_rows = []
    for line in lines[data_start:]:
        if line.strip() and not line.strip().startswith('#'):
            data_rows.append(line.strip().split())
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # Verify the image column exists
    if image_column_name not in df.columns:
        raise ValueError(f"Column '{image_column_name}' not found in star file")
    
    return df


def process_images(df, image_column, input_dir, output_stack_path, n_workers=1):
    """
    Process images in the order they appear in the star file,
    concatenate them, and save as a new stack.
    Returns updated DataFrame with new image references.
    """
    # Parse all image references up front
    image_refs = [parse_image_reference(ref) for ref in df[image_column]]

    # Resolve filepaths and group (out_idx, img_idx) by source file to open
    # each MRC only once instead of once per particle
    resolved = []
    file_groups = {}  # filepath -> [(out_idx, img_idx), ...]
    for out_idx, (img_idx, filename) in enumerate(image_refs):
        filepath = filename if os.path.exists(filename) else os.path.join(input_dir, os.path.basename(filename))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MRC file not found: {filepath}")
        resolved.append(filepath)
        file_groups.setdefault(filepath, []).append((out_idx, img_idx))

    n_images = len(image_refs)
    n_files = len(file_groups)
    logging.info(f"Found {n_images} particles across {n_files} source file(s)")

    # Determine 2-D image shape from the first file's header only
    with mrcfile.open(resolved[0], mode='r', permissive=True) as mrc:
        img_shape = mrc.data.shape[-2:]  # (ny, nx) for both 2-D and stack
    logging.info(f"Particle shape: {img_shape[0]}x{img_shape[1]} → output {output_stack_path}")

    # Sort entries within each file by source index so disk reads are
    # sequential (reduces seek time on spinning disks / slow NFS).
    file_items = list(file_groups.items())
    for _, entries in file_items:
        entries.sort(key=lambda x: x[1])

    def copy_file(filepath, entries, out_data):
        """Copy all particles from one source file into the output mmap.
        Uses chunked numpy fancy indexing to replace the per-particle Python
        loop, reducing loop overhead ~_COPY_CHUNK× while bounding peak RAM.
        Each thread works on non-overlapping output indices, so no locking is
        needed for the array writes.
        """
        out_arr = np.array([e[0] for e in entries], dtype=np.intp)
        src_arr = np.array([e[1] for e in entries], dtype=np.intp)
        with mrcfile.mmap(filepath, mode='r', permissive=True) as mrc_in:
            src = mrc_in.data
            if src.ndim == 3:
                if src_arr[-1] >= src.shape[0]:  # sorted → last element is max
                    raise IndexError(f"Image index {src_arr[-1]} out of bounds in {filepath}")
                for start in range(0, len(entries), _COPY_CHUNK):
                    sl = slice(start, start + _COPY_CHUNK)
                    out_data[out_arr[sl]] = src[src_arr[sl]].astype(np.float32)
            else:
                if src_arr[0] != 0:
                    raise IndexError(f"Image index {src_arr[0]} invalid for single image {filepath}")
                out_data[out_arr[0]] = src.astype(np.float32)
        return len(entries)

    effective_workers = min(n_workers, n_files)

    # Pre-allocate memory-mapped output — slices are written directly to disk,
    # so the full stack is never held in RAM.  mrc_mode 2 = float32.
    with mrcfile.new_mmap(output_stack_path, shape=(n_images, *img_shape),
                          mrc_mode=2, overwrite=True) as mrc_out:
        out_data = mrc_out.data
        with tqdm(total=n_images, unit="ptcl", desc="Writing stack") as pbar:
            if effective_workers <= 1:
                for filepath, entries in file_items:
                    pbar.set_postfix(file=os.path.basename(filepath), refresh=False)
                    pbar.update(copy_file(filepath, entries, out_data))
            else:
                # Each future reads a different source file and writes to
                # non-overlapping output indices, so array writes are safe
                # without a lock.  The lock only guards tqdm.
                lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    futures = {
                        pool.submit(copy_file, fp, entries, out_data): os.path.basename(fp)
                        for fp, entries in file_items
                    }
                    for fut in as_completed(futures):
                        n = fut.result()  # re-raises any exception from the thread
                        with lock:
                            pbar.update(n)

    # Update image references in DataFrame (1-based indices, as RELION requires)
    output_filename = os.path.basename(output_stack_path)
    df[image_column] = [f"{i+1:06d}@{output_filename}" for i in range(n_images)]

    return df


def save_star_file(df, output_path, original_star_path):
    """
    Save the updated DataFrame as a star file, preserving the original format.
    Handles RELION 3.1+ files with multiple data blocks (e.g. data_optics +
    data_particles) by locating the particles loop_ specifically, so the full
    original header (including any optics block) is preserved verbatim.
    """
    with open(original_star_path, 'r') as f:
        lines = f.readlines()

    # Locate the particles data block and the loop_ that follows it,
    # mirroring the same logic used in read_star_file.
    particles_block_found = False
    loop_start = None
    data_rows_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("data_particles", "data_"):
            particles_block_found = True
        elif stripped == "loop_" and particles_block_found:
            loop_start = i
        elif loop_start is not None and stripped and stripped[0] != '_':
            # First non-header, non-empty line after the column labels
            data_rows_start = i
            break

    if loop_start is None:
        raise ValueError("Could not find 'loop_' in original star file")
    if data_rows_start is None:
        raise ValueError("Could not find data rows in original star file")

    # Everything before the data rows (includes optics block, loop_, column
    # labels for the particles block) is written verbatim.
    with open(output_path, 'w') as f:
        f.writelines(lines[:data_rows_start])

        # Build all rows as one string and flush in a single syscall
        f.write('\n'.join(
            ' '.join(str(val) for val in row)
            for row in df.itertuples(index=False, name=None)
        ) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process Relion star files and combine MRC images.')
    parser.add_argument('--star_file', required=True, help='Input Relion star file')
    parser.add_argument('--input_dir', required=True, help='Directory containing MRC files')
    parser.add_argument('--output_stack', required=True, help='Output MRC stack file')
    parser.add_argument('--output_star', required=True, help='Output star file')
    parser.add_argument('--image_column', default='rlnImageName', help='Column name for image references')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel reader threads (default: 1). '
                             'Increase when particles span many source files '
                             'and storage supports concurrent reads.')

    args = parser.parse_args()
    
    logging.info(f"Reading star file: {args.star_file}")
    df = read_star_file(args.star_file, args.image_column)
    logging.info(f"{len(df)} particle rows loaded")

    logging.info(f"Input dir: {args.input_dir}")
    t0 = time.monotonic()
    df = process_images(df, args.image_column, args.input_dir, args.output_stack, n_workers=args.workers)
    elapsed = time.monotonic() - t0

    logging.info(f"Saving star file: {args.output_star}")
    save_star_file(df, args.output_star, args.star_file)

    stack_gb = os.path.getsize(args.output_stack) / 1e9
    print(
        f"\n--- Summary ---"
        f"\n  Particles written : {len(df):,}"
        f"\n  Output stack      : {args.output_stack} ({stack_gb:.2f} GB)"
        f"\n  Output star file  : {args.output_star}"
        f"\n  Elapsed           : {elapsed:.1f}s"
        f"\n  Throughput        : {len(df)/elapsed:,.0f} ptcl/s"
    )
    logging.info("Done")


if __name__ == "__main__":
    main()

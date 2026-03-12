import os
import argparse
import logging
import re
import time
import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def process_images(df, image_column, input_dir, output_stack_path):
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

    # Pre-allocate memory-mapped output — slices are written directly to disk,
    # so the full stack is never held in RAM.  mrc_mode 2 = float32.
    with mrcfile.new_mmap(output_stack_path, shape=(n_images, *img_shape),
                          mrc_mode=2, overwrite=True) as mrc_out:
        with tqdm(total=n_images, unit="ptcl", desc="Writing stack") as pbar:
            for filepath, entries in file_groups.items():
                pbar.set_postfix(file=os.path.basename(filepath), refresh=False)
                # Open each source file as memory-mapped for efficient slice access
                with mrcfile.mmap(filepath, mode='r', permissive=True) as mrc_in:
                    src = mrc_in.data
                    for out_idx, img_idx in entries:
                        if src.ndim == 3:
                            if img_idx >= src.shape[0]:
                                raise IndexError(f"Image index {img_idx} out of bounds in {filepath}")
                            mrc_out.data[out_idx] = src[img_idx].astype(np.float32)
                        else:
                            if img_idx != 0:
                                raise IndexError(f"Image index {img_idx} invalid for single image {filepath}")
                            mrc_out.data[out_idx] = src.astype(np.float32)
                        pbar.update()

    # Update image references in DataFrame (1-based indices, as RELION requires)
    output_filename = os.path.basename(output_stack_path)
    df[image_column] = [f"{i+1:06d}@{output_filename}" for i in range(n_images)]

    return df


def save_star_file(df, output_path, original_star_path):
    """
    Save the updated DataFrame as a star file, preserving the original format.
    """
    # Read the original file to get the header
    with open(original_star_path, 'r') as f:
        lines = f.readlines()
    
    header_lines = []
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "loop_":
            data_start = i
            header_lines = lines[:data_start+1]
            break
    
    if data_start is None:
        raise ValueError("Could not find 'loop_' in original star file")
    
    # Get column headers
    column_headers = []
    i = data_start + 1
    while i < len(lines) and lines[i].strip() and lines[i].strip()[0] == '_':
        column_headers.append(lines[i].strip())
        i += 1
    
    # Write the new star file
    with open(output_path, 'w') as f:
        # Write the header
        f.writelines(header_lines)
        
        # Write column headers
        for header in column_headers:
            f.write(f"{header}\n")
        
        # Write data — itertuples is ~100x faster than iterrows for large tables
        for row in df.itertuples(index=False, name=None):
            f.write(' '.join(str(val) for val in row) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process Relion star files and combine MRC images.')
    parser.add_argument('--star_file', required=True, help='Input Relion star file')
    parser.add_argument('--input_dir', required=True, help='Directory containing MRC files')
    parser.add_argument('--output_stack', required=True, help='Output MRC stack file')
    parser.add_argument('--output_star', required=True, help='Output star file')
    parser.add_argument('--image_column', default='rlnImageName', help='Column name for image references')
    
    args = parser.parse_args()
    
    logging.info(f"Reading star file: {args.star_file}")
    df = read_star_file(args.star_file, args.image_column)
    logging.info(f"{len(df)} particle rows loaded")

    logging.info(f"Input dir: {args.input_dir}")
    t0 = time.monotonic()
    df = process_images(df, args.image_column, args.input_dir, args.output_stack)
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

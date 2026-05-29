import os
import argparse
import re
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_image_reference(ref_string):
    """
    Parse Relion's image reference format: 000index@filename.mrc
    Returns (index, filename)
    """
    if match := re.match(r'(\d+)@(.+)', ref_string):
        return int(match[1]), match[2]
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
    
    logger = logging.getLogger(__name__)
    n_cols = len(column_names)
    data_rows = []
    for lineno, line in enumerate(lines[data_start:], start=data_start + 1):
        if line.strip() and not line.strip().startswith('#'):
            fields = line.strip().split()
            if len(fields) > n_cols:
                logger.warning(f"Line {lineno} has {len(fields)} fields (expected {n_cols}); "
                               f"extra fields truncated.")
                fields = fields[:n_cols]
            data_rows.append(fields)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # Verify the image column exists
    if image_column_name not in df.columns:
        raise ValueError(f"Column '{image_column_name}' not found in star file")
    
    return df


def find_mrc_file(filename, input_dirs, star_file_dir):
    """
    Resolve the full path to an MRC file.
    If input_dirs are provided, each is tried as a base directory.
    If none are provided (or none contain the file), fall back to
    using the star file's own directory as the base.
    """
    search_bases = list(input_dirs) if input_dirs else [
        star_file_dir,
        os.path.dirname(star_file_dir),  # RELION project dir (parent of star file dir)
    ]
    for base in search_bases:
        filepath = os.path.join(base, filename)
        if os.path.exists(filepath):
            return filepath
    tried = [os.path.join(b, filename) for b in search_bases]
    raise FileNotFoundError(
        f"MRC file not found: {filename}\nSearched in: {tried}"
    )


def _load_images_from_file(filepath, indices_needed):
    """
    Open a single MRC file once and extract all required frames.
    Uses memory-mapping so the OS page cache is shared across workers and
    only the accessed pages are loaded from disk.
    indices_needed: list of (orig_idx, array_idx) — array_idx is 0-based.
    Returns: list of (orig_idx, image_data) tuples.
    """
    results = []
    with mrcfile.mmap(filepath, mode='r', permissive=True) as mrc:
        for orig_idx, array_idx in indices_needed:
            if mrc.data.ndim == 3:
                if array_idx < 0 or array_idx >= mrc.data.shape[0]:
                    raise IndexError(
                        f"Image index {array_idx + 1} out of bounds for {filepath} "
                        f"(stack has {mrc.data.shape[0]} images)"
                    )
                # .copy() detaches the array from the mmap before the file closes
                results.append((orig_idx, mrc.data[array_idx].copy()))
            else:
                if array_idx != 0:
                    raise IndexError(
                        f"Image index {array_idx + 1} invalid for single-image file {filepath}"
                    )
                results.append((orig_idx, mrc.data.copy()))
    return results


def process_images(df, image_column, input_dirs, star_file_dir, output_stack_path, workers=1):
    """
    Load images referenced in the star file, concatenate them, and save as a
    new MRC stack.  Images are grouped by source file so each .mrcs is opened
    only once; files are processed in parallel when workers > 1.
    Returns updated DataFrame with new image references.
    """
    logger = logging.getLogger(__name__)

    # Parse image references (preserve original star-file order via orig_idx)
    image_refs = []
    for ref in df[image_column]:
        idx, filename = parse_image_reference(ref)
        image_refs.append((idx, filename))
    image_refs_sorted = sorted(enumerate(image_refs), key=lambda x: x[0])

    # Group by resolved filepath so each file is opened only once
    logger.info("Resolving file paths...")
    file_groups = defaultdict(list)  # filepath -> [(orig_idx, array_idx), ...]
    for orig_idx, (img_idx, filename) in image_refs_sorted:
        filepath = find_mrc_file(filename, input_dirs, star_file_dir)
        file_groups[filepath].append((orig_idx, img_idx - 1))  # convert to 0-based

    n_files = len(file_groups)
    n_images = len(image_refs_sorted)
    logger.info(f"Loading {n_images} images from {n_files} unique MRC file(s) "
                f"using {workers} worker(s)...")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_load_images_from_file, fp, indices): fp
            for fp, indices in file_groups.items()
        }
        with tqdm(total=n_images, desc="Loading images", unit="img") as pbar:
            for future in as_completed(futures):
                for orig_idx, image in future.result():
                    results[orig_idx] = image
                    pbar.update(1)

    logger.info("Stacking images...")
    image_stack = np.stack([results[i] for i in range(n_images)], axis=0)

    logger.info(f"Saving output stack: {output_stack_path}")
    with mrcfile.new(output_stack_path, overwrite=True) as mrc:
        mrc.set_data(image_stack.astype(np.float32))

    # Update image references in DataFrame (1-based RELION convention)
    output_filename = os.path.basename(output_stack_path)
    for i in range(len(df)):
        df.at[i, image_column] = f"{i + 1:06d}@{output_filename}"

    return df


def verify_output(output_star_path, output_stack_path, original_star_path,
                  image_column, input_dirs, star_file_dir, n_samples=50):
    """
    Sanity-check the output stack and star file:
      1. Indices in the output star are sequential and 1-based.
      2. Particle count matches between original and output star.
      3. For a random sample of particles, pixel data in the output stack
         matches the corresponding frame in the original source .mrcs.
    Raises AssertionError on any failure; returns True on success.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Running output verification ---")

    orig_df = read_star_file(original_star_path, image_column)
    out_df = read_star_file(output_star_path, image_column)

    # 1. Particle count
    assert len(orig_df) == len(out_df), (
        f"Particle count mismatch: original={len(orig_df)}, output={len(out_df)}"
    )
    logger.info(f"[PASS] Particle count: {len(out_df)}")

    # 2. Sequential 1-based indices in output star
    out_stack_name = os.path.basename(output_stack_path)
    for row_i, ref in enumerate(out_df[image_column], start=1):
        idx, fname = parse_image_reference(ref)
        assert idx == row_i, (
            f"Non-sequential index at row {row_i}: got {idx}"
        )
        assert os.path.basename(fname) == out_stack_name, (
            f"Row {row_i} points to '{fname}', expected '{out_stack_name}'"
        )
    logger.info("[PASS] Output star indices are sequential and 1-based")

    # 3. Pixel-data spot checks
    n_samples = min(n_samples, len(orig_df))
    sample_rows = random.sample(range(len(orig_df)), n_samples)
    logger.info(f"Spot-checking {n_samples} random particles against source files...")

    failures = []
    with mrcfile.mmap(output_stack_path, mode='r', permissive=True) as out_mrc:
        for row_i in tqdm(sample_rows, desc="Verifying", unit="particle"):
            # Expected frame from original source
            src_idx, src_file = parse_image_reference(orig_df[image_column].iloc[row_i])
            src_path = find_mrc_file(src_file, input_dirs, star_file_dir)
            with mrcfile.mmap(src_path, mode='r', permissive=True) as src_mrc:
                if src_mrc.data.ndim == 3:
                    src_frame = src_mrc.data[src_idx - 1].astype(np.float32)
                else:
                    src_frame = src_mrc.data.astype(np.float32)

            # Corresponding frame in output stack (1-based index from output star)
            out_idx, _ = parse_image_reference(out_df[image_column].iloc[row_i])
            out_frame = out_mrc.data[out_idx - 1].astype(np.float32)

            if not np.array_equal(src_frame, out_frame):
                failures.append(
                    f"Row {row_i + 1}: source {src_file}[{src_idx}] != "
                    f"output stack frame {out_idx}"
                )

    if failures:
        for msg in failures:
            logger.error(f"[FAIL] {msg}")
        raise AssertionError(
            f"{len(failures)}/{n_samples} spot-checks failed (see above)"
        )

    logger.info(f"[PASS] All {n_samples} pixel-data spot-checks passed")
    logger.info("--- Verification complete ---")
    return True


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
        
        # Write data
        for _, row in df.iterrows():
            f.write(' '.join(str(val) for val in row) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process Relion star files and combine MRC images.')
    parser.add_argument('--star_file', required=True, help='Input Relion star file')
    parser.add_argument('--input_dir', nargs='*', default=[],
                        help='Base director(y/ies) containing MRC files. '
                             'Can be specified as a space-separated list. '
                             'If omitted, paths are resolved relative to the star file location.')
    parser.add_argument('--output_stack', required=True, help='Output MRC stack file')
    parser.add_argument('--output_star', required=True, help='Output star file')
    parser.add_argument('--image_column', default='rlnImageName',
                        help='Column name for image references (default: rlnImageName)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='Number of parallel workers for image loading '
                             '(default: number of CPU cores)')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity (default: INFO)')
    parser.add_argument('--log_file', default=None,
                        help='Optional path to write log output to a file')
    parser.add_argument('--verify', action='store_true',
                        help='After writing outputs, run a sanity check comparing '
                             'a sample of output frames against their source files')
    parser.add_argument('--verify_samples', type=int, default=50,
                        help='Number of random particles to spot-check (default: 50)')

    args = parser.parse_args()

    # Configure logging
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    star_file_dir = os.path.dirname(os.path.abspath(args.star_file))
    if args.input_dir:
        logger.info(f"Using input directories: {args.input_dir}")
    else:
        logger.info(f"No --input_dir given; resolving paths relative to: {star_file_dir} "
                    f"(and its parent)")

    logger.info(f"Reading star file: {args.star_file}")
    df = read_star_file(args.star_file, args.image_column)
    logger.info(f"Found {len(df)} particles")

    df = process_images(df, args.image_column, args.input_dir, star_file_dir,
                        args.output_stack, workers=args.workers)

    logger.info(f"Saving updated star file: {args.output_star}")
    save_star_file(df, args.output_star, args.star_file)

    if args.verify:
        verify_output(
            output_star_path=args.output_star,
            output_stack_path=args.output_stack,
            original_star_path=args.star_file,
            image_column=args.image_column,
            input_dirs=args.input_dir,
            star_file_dir=star_file_dir,
            n_samples=args.verify_samples,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()

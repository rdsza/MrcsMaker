import os
import argparse
import re
import mrcfile
import numpy as np
import pandas as pd


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
    # Parse image references
    image_refs = []
    for ref in df[image_column]:
        idx, filename = parse_image_reference(ref)
        image_refs.append((idx, filename))
    
    # Sort by original order in star file
    image_refs_sorted = sorted(enumerate(image_refs), key=lambda x: x[0])
    
    # Load images
    images = []
    for orig_idx, (img_idx, filename) in image_refs_sorted:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MRC file not found: {filepath}")
        
        with mrcfile.open(filepath) as mrc:
            # Extract the specific image from the stack
            if mrc.data.ndim == 3:  # It's a stack
                if img_idx >= mrc.data.shape[0]:
                    raise IndexError(f"Image index {img_idx} out of bounds for file {filename}")
                images.append(mrc.data[img_idx].copy())
            else:  # It's a single image
                if img_idx != 0:
                    raise IndexError(f"Image index {img_idx} invalid for single image file {filename}")
                images.append(mrc.data.copy())
    
    # Stack images
    image_stack = np.stack(images, axis=0)
    
    # Save as new MRC stack
    with mrcfile.new(output_stack_path, overwrite=True) as mrc:
        mrc.set_data(image_stack.astype(np.float32))
    
    # Update image references in DataFrame
    output_filename = os.path.basename(output_stack_path)
    for i in range(len(df)):
        df.at[i, image_column] = f"{i:06d}@{output_filename}"
    
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
        
        # Write data
        for _, row in df.iterrows():
            f.write(' '.join(str(val) for val in row) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process Relion star files and combine MRC images.')
    parser.add_argument('--star_file', required=True, help='Input Relion star file')
    parser.add_argument('--input_dir', required=True, help='Directory containing MRC files')
    parser.add_argument('--output_stack', required=True, help='Output MRC stack file')
    parser.add_argument('--output_star', required=True, help='Output star file')
    parser.add_argument('--image_column', default='rlnImageName', help='Column name for image references')
    
    args = parser.parse_args()
    
    # Read star file
    print(f"Reading star file: {args.star_file}")
    df = read_star_file(args.star_file, args.image_column)
    
    # Process images
    print(f"Processing images from directory: {args.input_dir}")
    df = process_images(df, args.image_column, args.input_dir, args.output_stack)
    
    # Save updated star file
    print(f"Saving updated star file: {args.output_star}")
    save_star_file(df, args.output_star, args.star_file)
    
    print("Done!")


if __name__ == "__main__":
    main()

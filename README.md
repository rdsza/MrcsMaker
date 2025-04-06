# MrcsMaker
This repo contains a python script that takes as input a relion star file (containing particles data from various mrc stack files) and combines them into a single mrcs stack file and reindexes the old star file with new ImageName column.


### Functionality blocks
1. Parsing the star file: The script reads the Relion star file format, which has a specific structure with data blocks and column headers.
2. Extracting image references: It parses the image column (default: 'rlnImageName') which contains references in the format "000123@filename.mrc".
3. Loading MRC files: Using the mrcfile library, it loads each referenced image in the correct order.
4. Creating a combined stack: It concatenates all images into a single numpy array and saves it as a new MRC stack.
5. Updating references: It rewrites the image column with new indices pointing to the combined stack.

#### Usage:
```sh
python3 combine_mrc_stacks.py --star_file particles.star --input_dir ./mrc_files/ --output_stack combined.mrc --output_star updated.star
```

#### Logic behind implementation choices:
1. Using pandas for data handling: Provides a clean way to manipulate the star file data.
2. Regular expressions for parsing: The most flexible way to extract indices and filenames from Relion's reference format.
3. mrcfile library: This is a well maintained library specifically for MRC files that handles proper header information.
4. Preserving original star file format: Rather than just outputting a basic star file, the script attempts to preserve the original format and just update the relevant column.
5. In-memory processing: This approach loads all images into memory before saving. For very large datasets, you might need to modify this to process in batches.

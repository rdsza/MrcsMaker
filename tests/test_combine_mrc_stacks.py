import os
import pytest
import numpy as np
import mrcfile
import pandas as pd
from combine_mrc_stacks import parse_image_reference, read_star_file, process_images

def test_parse_image_reference():
    # Test valid reference
    idx, filename = parse_image_reference("000123@test.mrc")
    assert idx == 123
    assert filename == "test.mrc"
    
    # Test invalid reference
    with pytest.raises(ValueError):
        parse_image_reference("invalid_reference")

def test_read_star_file():
    # Create a temporary star file
    star_content = """data_particles
loop_
_rlnImageName
000001@test1.mrc
000002@test2.mrc
"""
    with open("tests/test_data/test.star", "w") as f:
        f.write(star_content)
    
    # Test reading the star file
    df = read_star_file("tests/test_data/test.star", "rlnImageName")
    assert len(df) == 2
    assert df.iloc[0]["rlnImageName"] == "000001@test1.mrc"
    assert df.iloc[1]["rlnImageName"] == "000002@test2.mrc"

def test_process_images():
    # Create test MRC files
    test_data = np.random.rand(2, 10, 10).astype(np.float32)
    
    # Create first MRC file
    with mrcfile.new("tests/test_data/test1.mrc", overwrite=True) as mrc:
        mrc.set_data(test_data[0])
    
    # Create second MRC file
    with mrcfile.new("tests/test_data/test2.mrc", overwrite=True) as mrc:
        mrc.set_data(test_data[1])
    
    # Create test DataFrame
    df = pd.DataFrame({
        "rlnImageName": ["000000@test1.mrc", "000000@test2.mrc"]
    })
    
    # Process images
    output_stack = "tests/test_data/output.mrc"
    updated_df = process_images(df, "rlnImageName", "tests/test_data", output_stack)
    
    # Verify output
    assert os.path.exists(output_stack)
    assert len(updated_df) == 2
    assert updated_df.iloc[0]["rlnImageName"] == "000000@output.mrc"
    assert updated_df.iloc[1]["rlnImageName"] == "000001@output.mrc"
    
    # Clean up
    os.remove("tests/test_data/test1.mrc")
    os.remove("tests/test_data/test2.mrc")
    os.remove(output_stack) 
from butterfree.data import metadata as md
import numpy as np

def test_get_cmd_metadata():

    metadata = md.get_cmd_metadata()
    assert len(metadata.columns_and_types) == 98
    assert 'gender' in metadata.columns_and_types
    i = 0
    for row in metadata:
        assert type(row['minimum_read_length'][0]) is np.float64
        i += 1
        if i == 10:
            break

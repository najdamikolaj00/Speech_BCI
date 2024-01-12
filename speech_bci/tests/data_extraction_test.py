# PATH = 'd:/SPEECH_BCI' # Where the repository was cloned
# import sys
# sys.path.append(PATH)

import numpy as np
from data_extraction import Extract_data_from_subject

def test_extract_data_from_subject(root_dir, datatype, N_S):

    # Extract data for the specified subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Test assertions
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(Y, np.ndarray), "Y should be a numpy array"
    assert X.ndim == 3, "X should be a 3-dimensional array"
    assert Y.ndim == 2, "Y should be a 2-dimensional array"


    print("Test passed: Data extraction is working as expected.")


if __name__ == "__main__":

    root_dir = "Data"
    datatype = "EEG"
    N_S = 1

    test_extract_data_from_subject(root_dir, datatype, N_S)
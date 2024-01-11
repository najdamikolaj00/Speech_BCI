"""
Utilitys from extract, read and load data from Inner Speech Dataset

@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar

Modified by:
@author_of_modifications: Mikołaj Najda

If any changes, write a brief description here and the author:

- utilities import is different due to environment (Mikołaj Najda)
- Extract_block_data_from_subject: variable "datatype" is now converted to lower case (Mikołaj Najda)
- Now the code is commented to keep a new user with the utilities (Mikołaj Najda)
- _aggregate_data is and additional, helper function to make the code more readable (Mikołaj Najda)

"""
# PATH = 'd:/SPEECH_BCI' # Where the repository was cloned
# import sys
# sys.path.append(PATH)

import mne
import gc
import pickle
import numpy as np

from utilities import sub_name, unify_names

def Extract_subject_from_BDF(root_dir, N_S, N_B):
    """
    Extract EEG data from a BDF file for a specific subject and session block.

    Parameters:
    - root_dir: The root directory where BDF files are stored.
    - N_S: Subject number (integer).
    - N_B: Block number (integer).

    Returns:
    - rawdata: The loaded raw EEG data.
    - Num_s: The subject number, formatted as a string.
    """
    # Correct subject number format if necessary
    Num_s = sub_name(N_S)

    # Construct file path and load EEG data
    file_path = f"{root_dir}/{Num_s}/ses-0{N_B}/eeg/{Num_s}_ses-0{N_B}_task-innerspeech_eeg.bdf"
    rawdata = mne.io.read_raw_bdf(input_fname=file_path, preload=True, verbose='WARNING')
    return rawdata, Num_s

def Extract_data_from_subject(root_dir, N_S, datatype):
    """
    Load all EEG data blocks for a single subject and compile the data.

    Parameters:
    - root_dir: The root directory where EEG data files are stored.
    - N_S: Subject number (integer).
    - datatype: Type of data to be extracted (e.g., 'eeg', 'exg', 'baseline').

    Returns:
    - X: Combined EEG data from all blocks.
    - Y: Combined event data from all blocks.
    """
    # Initialize data containers
    data = {}
    y = {}
    N_B_arr = [1, 2, 3]
    datatype = datatype.lower()

    # Process each block
    for N_B in N_B_arr:
        Num_s = sub_name(N_S)
        y[N_B] = load_events(root_dir, N_S, N_B)

        # Construct file path based on datatype and load data
        file_path = f"{root_dir}/derivatives/{Num_s}/ses-0{N_B}/{Num_s}_ses-0{N_B}_{datatype}-epo.fif"
        if datatype in ["eeg", "exg", "baseline"]:
            X = mne.read_epochs(file_path, verbose='WARNING')
            data[N_B] = X._data
        else:
            raise Exception("Invalid datatype")

    # Combine data from all blocks
    X = np.vstack([data.get(key) for key in sorted(data.keys())])
    Y = np.vstack([y.get(key) for key in sorted(y.keys())])

    return X, Y

def Extract_block_data_from_subject(root_dir, N_S, datatype, N_B):
    """
    Loads a specific block of data for a given subject and datatype.

    Parameters:
    - root_dir: The root directory where data files are stored.
    - N_S: Subject number (integer).
    - datatype: Type of data to extract ('eeg', 'exg', 'baseline').
    - N_B: Block number (integer).

    Returns:
    - X: Data for the specified block and datatype.
    - Y: Event data associated with the specified block.
    """
    # Correct the subject number format and convert datatype to lowercase
    Num_s = sub_name(N_S)
    datatype = datatype.lower()
    
    # Load event data
    Y = load_events(root_dir, N_S, N_B)

    # Construct file path for the specified datatype and load data
    sub_dir = f"{root_dir}/derivatives/{Num_s}/ses-0{N_B}/{Num_s}_ses-0{N_B}"
    file_name = f"{sub_dir}_{datatype}-epo.fif"
    if datatype in ["eeg", "exg", "baseline"]:
        X = mne.read_epochs(file_name, verbose='WARNING')
    else:
        raise Exception("Invalid Datatype")
    
    return X, Y

def Extract_report(root_dir, N_B, N_S):
    """
    Extracts a stored report for a specific subject and block.

    Parameters:
    - root_dir: The root directory where report files are stored.
    - N_B: Block number (integer).
    - N_S: Subject number (integer).

    Returns:
    - report: The loaded report object.
    """
    # Correct the subject number format
    Num_s = sub_name(N_S)

    # Construct file path and load the report
    file_name = f"{root_dir}/derivatives/{Num_s}/ses-0{N_B}/{Num_s}_ses-0{N_B}_report.pkl"
    with open(file_name, 'rb') as input_file:
        report = pickle.load(input_file)

    return report

def Extract_TFR(TRF_dir, Cond, Class, TFR_method, TRF_type):
    """
    Extracts time-frequency representation (TFR) data for given conditions.

    Parameters:
    - TRF_dir: Directory where TFR files are stored.
    - Cond: Specific condition to filter.
    - Class: Specific class to filter.
    - TFR_method: The method used for TFR calculation.
    - TRF_type: The type of TFR data to extract.

    Returns:
    - TRF: Time-frequency representation data.
    """
    # Unify the names for consistency
    Cond, Class = unify_names(Cond, Class)
    
    # Construct the file path and load the TFR data
    fname = f"{TRF_dir}{TFR_method}_{Cond}_{Class}_{TRF_type}-tfr.h5"
    TRF = mne.time_frequency.read_tfrs(fname)[0]

    return TRF

def Extract_data_multisubject(root_dir, N_S_list, datatype='EEG'):
    """
    Aggregates EEG data from multiple subjects and sessions.

    Parameters:
    - root_dir: The root directory where data files are stored.
    - N_S_list: List of subject numbers to process.
    - datatype: Type of data to extract (default is 'EEG').

    Returns:
    - X: Aggregated data across all subjects and sessions.
    - Y: Aggregated event data corresponding to X.
    """
    N_B_arr = [1, 2, 3]  # Array of session blocks
    tmp_list_X = []  # Temporary list to store data
    tmp_list_Y = []  # Temporary list to store event data
    rows = []  # List to store the number of rows in each dataset
    total_elem = len(N_S_list) * 3  # Total number of elements to process
    S = 0  # Subject counter
    datatype = datatype.lower()

    for N_S in N_S_list:
        print(f"Iteration {S}, Subject {N_S}")
        for N_B in N_B_arr:
            Num_s = sub_name(N_S)

            # Construct file names for data and events
            base_file_name = f"{root_dir}/derivatives/{Num_s}/ses-0{N_B}/{Num_s}_ses-0{N_B}"
            events_file_name = f"{base_file_name}_events.dat"
            data_tmp_Y = np.load(events_file_name, allow_pickle=True)
            tmp_list_Y.append(data_tmp_Y)

            # Load data based on datatype
            if datatype in ["eeg", "exg", "baseline"]:
                file_name = f"{base_file_name}_{datatype}-epo.fif"
                data_tmp_X = mne.read_epochs(file_name, verbose='WARNING')._data
                rows.append(data_tmp_X.shape[0])
                if S == 0 and N_B == 1:  # Set dimensions based on first subject and block
                    chann = data_tmp_X.shape[1]
                    steps = data_tmp_X.shape[2]
                    columns = data_tmp_Y.shape[1]
                tmp_list_X.append(data_tmp_X)
            else:
                raise Exception("Invalid Datatype")

            S += 1

    # Aggregate the data into a single array
    X, Y = _aggregate_data(tmp_list_X, tmp_list_Y, rows, chann, steps, columns, datatype)
    return X, Y

def _aggregate_data(tmp_list_X, tmp_list_Y, rows, chann, steps, columns, datatype):
    """
    Helper function to aggregate data into a single array.

    Parameters:
    - tmp_list_X: Temporary list containing data arrays.
    - tmp_list_Y: Temporary list containing event arrays.
    - rows: List containing the number of rows for each data array.
    - chann, steps, columns: Dimensions of the data arrays.
    - datatype: Type of data being processed.

    Returns:
    - X: Aggregated data array.
    - Y: Aggregated event array.
    """
    X = np.empty((sum(rows), chann, steps))
    Y = np.empty((sum(rows), columns))
    offset = 0

    for i in range(len(tmp_list_X)):
        X[offset:offset+rows[i], :, :] = tmp_list_X[i]
        if datatype in ["eeg", "exg"]:
            Y[offset:offset+rows[i], :] = tmp_list_Y[i]
        offset += rows[i]

    gc.collect()  # Clear memory
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y

def load_events(root_dir, N_S, N_B):
    """
    Loads event data for a specific subject and block.

    Parameters:
    - root_dir: The root directory where event files are stored.
    - N_S: Subject number (integer).
    - N_B: Block number (integer).

    Returns:
    - events: The loaded event data.
    """
    Num_s = sub_name(N_S)  # Correct the subject number format

    # Construct file path and load events
    file_name = f"{root_dir}/derivatives/{Num_s}/ses-0{N_B}/{Num_s}_ses-0{N_B}_events.dat"
    events = np.load(file_name, allow_pickle=True)
    
    return events
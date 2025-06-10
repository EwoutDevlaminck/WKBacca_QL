# Functions used for WKBacca automatic setup

# Import the necessary modules
import os
import sys
import numpy as np


def create_workspace(calc_path, new_calc_folder, folders=['input', 'output', 'config']):
    """
    Function to create a new workspace for the WKBeam calculation.

    Parameters:
    -----------
    calc_path : str
        The path to the main folder containing the WKBeam calculations.
    new_calc_folder : str
        The name of the new folder to be created.

    Returns:
    --------
    None
    """
    new_calc_path = os.path.join(calc_path, new_calc_folder)

    # Make the new folder
    if not os.path.exists(new_calc_path):
        os.makedirs(new_calc_path)
        print(f'New folder {new_calc_folder} created for the WKBeam calculation at {new_calc_path}')
    else:
        print(f'Folder {new_calc_folder} already exists at {new_calc_path}')

    # Make the subfolders

    for folder in folders:
        folder_path = os.path.join(new_calc_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f'Subfolder {folder} created at {folder_path}')
        else:
            print(f'Subfolder {folder} already exists at {folder_path}')

    print('Input data should be placed in the new input folder.')

    return None




def update_config(standard_file, new_values_file, output_file):
    with open(standard_file, 'r') as f:
        standard_lines = f.readlines()
    
    with open(new_values_file, 'r') as f:
        new_lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    new_values = {line.split('=')[0].strip(): line for line in new_lines if '=' in line}
    
    updated_lines = []
    for line in standard_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=')[0].strip()
            if key in new_values:
                updated_lines.append(new_values[key] + '\n')
                continue
        updated_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(updated_lines)

    return None


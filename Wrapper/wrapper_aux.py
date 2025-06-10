import os
import subprocess
import numpy as np


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

def create_dir(directory, subdirs=None):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    if subdirs:
        for subdir in subdirs:
            subpath = os.path.join(directory, subdir)
            if not os.path.exists(subpath):
                os.makedirs(subpath)
                print(f'Created directory: {subpath}')
                
    return None

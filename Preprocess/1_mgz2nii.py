#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

# Configuration for path control
MGZ_FILES_TO_CONVERT = ["brain.mgz", "brainmask.mgz"]
FS_MRI_RELATIVE_PATH = os.path.join("fs", "mri")

def convert_mgz_to_nii(input_dir, verbose=False):
    """
    Convert MGZ files to NIfTI format within the specified directory structure
    
    Parameters:
    input_dir (str): Path to the input directory containing subject folders
    verbose (bool): Whether to display detailed output
    
    Expected File Structure:
    input_dir/
    ├── subject1/
    │   └── fs/
    │       └── mri/
    │           ├── brain.mgz
    │           └── brainmask.mgz
    └── ...

    Usage Example:
       python 1_mgz2nii.py /path/to/subjects_root -v
    """
    # Validate input directory existence
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist")
        return
    
    # Get all subject directories
    subj_dirs = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    if not subj_dirs:
        print(f"Warning: No subfolders found in directory '{input_dir}'")
        return
    
    # Conversion counters
    converted_count = 0
    skipped_count = 0
    
    # Process each subject directory
    for subj in subj_dirs:
        subj_path = os.path.join(input_dir, subj)
        fs_mri_path = os.path.join(subj_path, FS_MRI_RELATIVE_PATH)
        
        # Check if fs/mri directory exists
        if not os.path.exists(fs_mri_path):
            if verbose:
                print(f"Skipping {subj}: {fs_mri_path} not found")
            skipped_count += 1
            continue
        
        # Find and convert specified MGZ files
        for mgz_file in MGZ_FILES_TO_CONVERT:
            mgz_path = os.path.join(fs_mri_path, mgz_file)
            if os.path.exists(mgz_path):
                # Construct output file path
                nii_file = mgz_file.replace(".mgz", ".nii.gz")
                nii_path = os.path.join(fs_mri_path, nii_file)
                
                # Check if output file already exists
                if os.path.exists(nii_path):
                    if verbose:
                        print(f"Skipping {mgz_file}: {nii_file} already exists")
                    continue
                
                # Execute conversion
                try:
                    cmd = ["mri_convert", mgz_path, nii_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    if verbose or result.returncode != 0:
                        print(f"Converted {subj}/{mgz_file} to {nii_file}")
                        if result.stdout:
                            print(f"Standard output: {result.stdout.strip()}")
                        if result.stderr:
                            print(f"Error output: {result.stderr.strip()}")
                    
                    converted_count += 1
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to convert {mgz_path}: {e.stderr.strip()}")
                except Exception as e:
                    print(f"Error: Command execution failed: {str(e)}")
            else:
                if verbose:
                    print(f"{mgz_file} not found in {subj}")
                skipped_count += 1
    
    # Output statistics
    print(f"\nConversion completed!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Skipped: {skipped_count} files")

def main():
    """Main function to handle command-line arguments and invoke conversion"""
    parser = argparse.ArgumentParser(description='Convert FreeSurfer MGZ files to NIfTI format')
    parser.add_argument('inputDIR', help='Root directory containing subject folders')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Normalize directory path
    input_dir = os.path.abspath(args.inputDIR)
    
    # Execute conversion
    convert_mgz_to_nii(input_dir, args.verbose)

if __name__ == "__main__":
    main()

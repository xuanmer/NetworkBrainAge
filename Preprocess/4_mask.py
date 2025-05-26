import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Configuration for path control (using environment variables for portability)
BRAIN_2MM_BASE_DIR = os.environ.get('BRAIN_2MM_DIR', os.path.join('data', 'brain_2mm', 'NC'))
YEO_BASE_DIR = os.environ.get('YEO_DIR', os.path.join('data', 'NC'))

# Expected File Structure (for GitHub repository):
# repository_root/
# ├── data/
# │   ├── brain_2mm/
# │   │   └── NC/
# │   │       ├── subject1/
# │   │       │   └── brain_2mm/
# │   │       │       └── brain_resample.nii.gz
# │   │       └── subject2/
# │   │           └── ...
# │   └── NC/
# │       ├── subject1/
# │       │   └── yeo_dwi_2mm/
# │       │       └── Yeo2011_7Networks_..._resample.nii.gz
# │       └── subject2/
# │           └── ...
# └── scripts/
#     └── extract_subnetworks.py

# Usage (for GitHub users):
# 1. Clone the repository
# 2. Organize data under `data/` directory as per the expected structure
# 3. Run with environment variables:
#    export BRAIN_2MM_DIR=./data/brain_2mm/NC
#    export YEO_DIR=./data/NC
#    python scripts/extract_subnetworks.py
# 4. Or use default paths by ensuring data is in the `data/` directory

# Get all subject IDs
subject_ids = [d for d in os.listdir(BRAIN_2MM_BASE_DIR) if os.path.isdir(os.path.join(BRAIN_2MM_BASE_DIR, d))]

# Process each subject
for subject_id in tqdm(subject_ids, desc="Processing subjects"):
    # Construct input file paths
    brain_file = os.path.join(BRAIN_2MM_BASE_DIR, subject_id, 'brain_2mm', 'brain_resample.nii.gz')
    yeo_file = os.path.join(YEO_BASE_DIR, subject_id, 'yeo_dwi_2mm', 
                           'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_to_dwi_resample.nii.gz')
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(brain_file):
        missing_files.append(brain_file)
    if not os.path.exists(yeo_file):
        missing_files.append(yeo_file)
    
    # Skip subject if any files are missing
    if missing_files:
        print(f"Warning: Subject {subject_id} is missing the following files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        continue
    
    # Get output directory (same as input brain file directory)
    output_dir = os.path.dirname(brain_file)
    
    # Load image data
    try:
        # Load brain image
        brain_img = nib.load(brain_file)
        brain_data = brain_img.get_fdata()
        
        # Load Yeo network parcellation
        yeo_img = nib.load(yeo_file)
        yeo_data = yeo_img.get_fdata()
        
        # Verify image dimensions match
        if brain_data.shape != yeo_data.shape:
            print(f"Warning: Image dimensions do not match for subject {subject_id}:")
            print(f"  - Brain dimensions: {brain_data.shape}")
            print(f"  - Yeo template dimensions: {yeo_data.shape}")
            continue
        
        # Extract 7 subnetworks
        for network_id in range(1, 8):
            # Create binary mask for current network
            mask = (yeo_data == network_id).astype(np.float32)
            
            # Apply mask to brain data
            masked_brain = brain_data * mask
            
            # Save result
            output_file = os.path.join(output_dir, f'brain_resample_subnet{network_id}.nii.gz')
            nib.save(nib.Nifti1Image(masked_brain, brain_img.affine, brain_img.header), output_file)
    
    except Exception as e:
        print(f"Error: Failed to process subject {subject_id}: {str(e)}")

print("All subjects processed!")

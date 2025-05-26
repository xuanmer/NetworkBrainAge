from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Workflow
import nipype.interfaces.io as nio
from nipype.interfaces import afni
import random
import sys
import os.path

BASE_DIR = '/path/to/subjects'
SUBJECT_GROUP = 'AD' 
TMP_DIR_TEMPLATE = os.path.join(BASE_DIR, 'tmp%s') 

# Dynamically constructed paths
SUBJECTS_DIR = os.path.join(BASE_DIR, SUBJECT_GROUP) # /path/to/subjects/AD
OUTPUT_DIR = os.path.join(SUBJECTS_DIR, 'brain_2mm')  

def registration(sub):
    """
    Register and resample the brain image to 2mm isotropic resolution
    
    Parameters:
    sub (str): Subject ID
    """
    # Create a temporary working directory with random name
    work_dir = TMP_DIR_TEMPLATE.format(''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 8)))
    
    # Input file path
    input_file = SUBJECTS_DIR / sub / "fs/mri/brain.nii.gz"
    
    # Create workflow
    workflow = Workflow(name='resample_brain', base_dir=str(work_dir))
    
    # Resample node
    resample = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', voxel_size=(2, 2, 2)),
        name='resample'
    )
    resample.inputs.in_file = str(input_file)
    
    # Datasink node
    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = str(OUTPUT_DIR)
    datasink.inputs.container = sub  # Store results in subject-specific subdirectory
    
    # Connect nodes
    workflow.connect(resample, 'out_file', datasink, 'brain_2mm')
    
    # Execute workflow
    workflow.run()
    
    # Cleanup temporary directory
    import shutil
    if work_dir.exists():
        shutil.rmtree(work_dir)

def start_batch_job(batch_file):
    """
    Start batch processing for subjects listed in a file
    
    Parameters:
    batch_file (str): Path to file containing subject IDs
    """
    with open(batch_file, "r") as f:
        subject_list = [line.strip() for line in f.readlines()]
        
        for subject in subject_list:
            # Check if output already exists
            output_path = OUTPUT_DIR / subject / f"{subject}_brain_resample.nii.gz"
            if output_path.exists():
                print(f"Skipping {subject}: Output already exists")
                continue
                
            # Process subject
            registration(subject)

if __name__ == '__main__':
    # Get batch file from command line argument
    batch_file = sys.argv[1]
    start_batch_job(batch_file)

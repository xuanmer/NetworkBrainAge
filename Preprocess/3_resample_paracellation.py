from nipype.pipeline import engine as pe
from qsiprep.engine import Workflow
import nipype.interfaces.io as nio
from nipype.interfaces import afni
import random
import sys
import os.path

def registration(sub):
    work_dir='/brain-age/tmp%s' % ''.join(random.sample('abcdefghijklmnopqrestuvwxyz0123456789',8)) #tmp directory
    inputs='/data_1mm/'+sub+'/brain.nii.gz' #input file, i.e. brain.nii.gz
    output_dir='/data_2mm/'+sub+'/' # output directory

    workflow = Workflow(name='resample_brain',base_dir=work_dir)
    resample=pe.Node(afni.Resample(outputtype='NIFTI_GZ',voxel_size=(2,2,2)),name='resample')
    resample.inputs.in_file=inputs
    datasink=pe.Node(interface=nio.DataSink(),name="datasink")
    datasink.inputs.base_directory=output_dir
    workflow.connect(resample,'out_file',datasink,'brain_2mm')
    workflow.run()
    os.system("rm -rf "+work_dir)

def startBatchJob(batchfile):
    with open(batchfile,"r") as f:
        sublist=list(map(lambda x:x.strip("\n"),f.readlines()))
        for sub in sublist:
            if os.path.exists("/data_2mm/"+sub+"/brain_2mm/brain_resample.nii.gz"):
                print("exist...")
                continue
            registration(sub)

if __name__=='__main__':
    batchfile=sys.argv[1]
    startBatchJob(batchfile)


import os.path
import random
# from qsiprep.niworkflows.interfaces.registration import RobustMNINormalizationRPT
from qsiprep.interfaces.niworkflows import RobustMNINormalizationRPT
from pkg_resources import resource_filename as pkgr
from nipype.pipeline import engine as pe
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.engine import Workflow
import nipype.interfaces.io as nio
from nipype.interfaces import afni
import sys

def _get_resampled(atlas_configs, atlas_name, to_retrieve):
    return atlas_configs[atlas_name][to_retrieve]

def registrationJob(sub):
    ref_img_brain = pkgr('qsiprep', 'data/mni_1mm_t1w_lps_brain.nii.gz')

    t1_2_mni = pe.Node(RobustMNINormalizationRPT(float=True, generate_report=True, flavor='precise', ), name='t1_2_mni',n_procs=4, mem_gb=2)
    t1_2_mni.inputs.template = 'MNI152NLin2009cAsym'
    t1_2_mni.inputs.reference_image = ref_img_brain
    t1_2_mni.inputs.orientation = "LPS"
    t1_2_mni.inputs.moving_image = '/data_1mm/' + sub + '/brain.nii.gz'  # Freesurfer生成的brain.mgz,
    t1_2_mni.inputs.moving_mask = '/data_1mm/' + sub + '/brainmask.nii.gz'  # Freesurfer生成的brainmask.mgz,

    atlas = ['yeo']
    print(atlas[0])
    space = 'T1w'
    get_atlas = pe.Node(GetConnectivityAtlases(space=space, atlas_names=atlas), name='get_atlas')

    get_atlas.inputs.reference_image='/data_1mm/'+sub+'/brain.nii.gz' # 1mm的brain.nii.gz

    ramdom_str=random.sample('abcdefghijklmnopqrestuvwxyz0123456789',8)
    work_dir='/brain/shulab/tmp%s' % ''.join(ramdom_str)
    workflow = Workflow(name='indi_atlas',base_dir=work_dir)
    workflow.connect([(t1_2_mni, get_atlas, [('inverse_composite_transform', 'forward_transform')])])
    resample = pe.Node(afni.Resample(outputtype='NIFTI_GZ', voxel_size=(2.0, 2.0, 2.0)), name='resample')
    workflow.connect(get_atlas, ('atlas_configs', _get_resampled, atlas[0], 'dwi_resolution_file'), resample, 'in_file')
    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = '/data_1mm/'+sub # output directory
    workflow.connect(t1_2_mni, 'composite_transform', datasink, 't12mni_warp')
    workflow.connect(resample, 'out_file', datasink, 'yeo_dwi_2mm')

    workflow.run()
    os.system("rm -rf "+work_dir)

def test_ants(batch):
    with open(batch,"r") as f:
        sublist=list(map(lambda x:x.strip("\n"),f.readlines()))
        for sub in sublist:
            registrationJob(sub)

if __name__=='__main__':
    subs=sys.argv[1]
    with open(subs,"r") as f:
        sublist=list(map(lambda x:x.strip("\n"),f.readlines()))
        for sub in sublist:
            if os.path.exists("/data_1mm/"+sub+"/yeo_dwi_2mm/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_to_dwi_resample.nii.gz"):
                print("exist........")
                continue
            registrationJob(sub)

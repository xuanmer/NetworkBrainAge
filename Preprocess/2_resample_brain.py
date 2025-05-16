from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Workflow
import nipype.interfaces.io as nio
from nipype.interfaces import afni
import random
import sys
import os.path

# 声明所有路径变量
BASE_DIR = '/home/shulab/bty/brain-age/zhe2'
SUBJECT_GROUP = 'AD'  # 可修改的受试者组名称
TMP_DIR_TEMPLATE = os.path.join(BASE_DIR, 'tmp%s')

# 动态构建的路径
SUBJECTS_DIR = os.path.join(BASE_DIR, SUBJECT_GROUP)
OUTPUT_DIR = os.path.join(SUBJECTS_DIR, 'brain_2mm')  # 改为基于 SUBJECTS_DIR

def registration(sub):
    work_dir = TMP_DIR_TEMPLATE % ''.join(random.sample('abcdefghijklmnopqrestuvwxyz0123456789', 8))  # tmp directory
    # 构建输入路径
    inputs = os.path.join(SUBJECTS_DIR, sub, 'fs/mri/brain.nii.gz')

    workflow = Workflow(name='resample_brain', base_dir=work_dir)
    resample = pe.Node(afni.Resample(outputtype='NIFTI_GZ', voxel_size=(2, 2, 2)), name='resample')
    resample.inputs.in_file = inputs
    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = OUTPUT_DIR
    # 设置 container 属性，确保每个受试者的结果保存到独立的子目录
    datasink.inputs.container = sub
    workflow.connect(resample, 'out_file', datasink, 'brain_2mm')
    workflow.run()
    os.system(f"rm -rf {work_dir}")

def startBatchJob(batchfile):
    with open(batchfile, "r") as f:
        sublist = list(map(lambda x: x.strip("\n"), f.readlines()))
        for sub in sublist:
            # 构建输出路径
            output_path = os.path.join(OUTPUT_DIR, sub, f'{sub}_brain_resample.nii.gz')
            if os.path.exists(output_path):
                print("exist...")
                continue
            registration(sub)

if __name__ == '__main__':
    batchfile = sys.argv[1]
    startBatchJob(batchfile)
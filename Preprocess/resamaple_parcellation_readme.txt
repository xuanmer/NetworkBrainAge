docker run --rm --entrypoint python \
-v {path to atlas}/:/atlas/qsirecon_atlases \
-v {path to site}/{group}/code:/code \
-v {path to site}/{group}/:/data_1mm \
pennbbl/qsiprep:0.19.1 \
/code/3_resample_yeo7.py /code/{site}_{group}_subjects.txt   

- Added atlas_config.json and Yeo7 atlas file for automated loading
- Applied RobustMNINormalizationRPT for template-to-T1 registration
- Used AFNI Resample to downsample registered template to 2mm resolution

{path to atlas}
atlas/
├── atlas_config.json
└── Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii

{path to site}/{group}/{sub}
└── fs
    └── mri
        ├── brainmask.mgz
        ├── brainmask.nii.gz
        ├── brain.mgz
        └── brain.nii.gz

{path to site}/{group}/code
code
├── 1_mgz2nii.py
├── 2_resample_brain.py
├── 3_resample_yeo7.py
├── 4_mask.py
└── {site}_{group}_subjects.txt

docker run --rm --entrypoint python \
-v /home/shulab/bty/brain-age/atlas/:/atlas/qsirecon_atlases \
-v /home/shulab/bty/brain-age/{site}/code:/code \
-v /home/shulab/bty/brain-age/{site}/{group}/:/data_1mm \
pennbbl/qsiprep:0.19.1 \
/code/3_resample_yeo7.py /code/{site}_{group}_subjects.txt   

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
└── {site}_{group}_subjects.txt



# NetworkBrainAge

A deep learning toolkit for predicting **brain age** from T1-weighted MRI scans using a Simple Fully Convolutional Network (SFCN) and network-wise analysis.

---

## Requirements

- Python 3.6+  
- TensorFlow 2.x (with Keras)  
- NumPy  
- Pandas  
- scikit-learn  
- nibabel  
- SciPy  
- Matplotlib  
---

## Data Preparation
1.	Organize your raw T1 NIfTI files in a folder, e.g.:
```bash
raw_data/
  ├── subject001_T1.nii.gz
  ├── subject002_T1.nii.gz
  └── ...
```
2. Run the preprocessing pipeline:
```bash
python Preprocess/preprocess_data.py \
  --input_dir raw_data/ \
  --output_dir processed_data/ \
  --normalize --skull_strip
```
3. Prepare a CSV file (labels.csv) with columns: subject_id, age, gender (and any covariates).

## Training
```bash
python training_SFCN.py \
```
Training logs (loss, MAE, R²) will be written to logs/SFCN/ for visualization in TensorBoard.

## Inference
```bash
python Network-wiseBrainAgePrediction.py
```
---
## License

This project is licensed under the MIT License. See LICENSE for details.

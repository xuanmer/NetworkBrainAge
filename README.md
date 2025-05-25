# NetworkBrainAge

A deep learning toolkit for predicting **brain age** from T1-weighted MRI scans using a Simple Fully Convolutional Network (SFCN) and network-wise analysis.

---

## Features

- **Data preprocessing**  
  – Scripts to normalize, skull-strip, and resample raw NIfTI images (in the `Preprocess/` folder).  
- **Data loading**  
  – `DataLoader.py` implements a Keras/TensorFlow-compatible data generator, with on-the-fly augmentation.  
- **Model definitions**  
  – `SFCN.py` defines the SFCN architecture for volumetric age regression.  
  – Additional architectures or graph-based models can be placed in the `Models/` folder.  
- **Training & evaluation**  
  – `training_SFCN.py` — train the SFCN model with early stopping, TensorBoard logging, and MAE/R² metrics.  
  – Built-in support for gender or scanner covariates.  
- **Inference**  
  – `predict.py` — load a saved model and run brain age prediction on new subjects.  
- **Network-wise analysis**  
  – `Network-wiseBrainAgePrediction.py` — aggregate regional/network predictions for group-level analysis.  

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

## Installation
1. Clone the repository
```bash
git clone https://github.com/xuanmer/NetworkBrainAge.git
cd NetworkBrainAge
```
2. Install dependencies
```
pip install -r requirements.txt
```
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
  --data_dir processed_data/ \
  --labels labels.csv \
  --batch_size 8 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --log_dir logs/SFCN/
```
- --data_dir: directory of preprocessed NIfTI volumes
- --labels: CSV with true ages (and optional covariates)
- --batch_size, --epochs, --learning_rate: training hyperparameters

Training logs (loss, MAE, R²) will be written to logs/SFCN/ for visualization in TensorBoard.

## Inference
```bash
python predict.py \
  --model_path logs/SFCN/best_model.h5 \
  --input_dir processed_data/ \
  --output_file predictions.csv
```
- --model_path: path to the saved .h5 model
- --input_dir: folder of preprocessed volumes
- --output_file: CSV to write subject IDs and predicted ages

---

## Network-wise Brain Age Analysis
To analyze network-specific age deviations:
```bash
python Network-wiseBrainAgePrediction.py \
  --features_dir extracted_features/ \
  --labels labels.csv \
  --output_dir network_results/ \
  --networks parcellation.yaml
```
This will compute age gaps per network/region and save summary statistics and plots in network_results/.

## License

This project is licensed under the MIT License. See LICENSE for details.
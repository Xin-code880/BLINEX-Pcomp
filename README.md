Dateset
You can download the datasets from the following link:
https://drive.google.com/drive/folders/18FUMDbciJ-16tpWCr0XJ32XvsvWzraV8?usp=drive_link

Run the following command to start training:
python main.py

Key Arguments
-ds: Specify the dataset (default: usps).
-uci: Whether to use UCI datasets (1: Yes, 0: No). Default = 1.
-prior: Class prior probability of positive samples. Default = 0.5.
-gpu: GPU ID to be used. Default = 0.
-m: Number of views. Default = 1.
-n: Number of unlabeled data pairs. Default = 1000.
-g: Gamma parameter for the Gaussian kernel (default: 1/n).

Code Structure
main.py — Entry point for running experiments.
data_composed.py — Data preparation and preprocessing.
model.py — Loss function and optimization.
predict.py — Evaluation and prediction.

# Binary-Classification-with-Neural-Network
This repository contains code to train a binary classification model using a neural network. The model is trained on a dataset, evaluated on a validation set, and used to predict labels on a test set. 

## Requirements

- Python 3.9
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn

## How to Use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ShiQiaoL/Binary-Classification-with-Neural-Network.git
   cd Binary-Classification-with-Neural-Network
   ```

2. **Download the data**:  
   Place the following CSV files in the repository:
   - `hetero_lattice_Allen_010_diff_06_20240514_tmds.csv` (Training data)
   - `hetero_lattice_Allen_010_diff_06_20221205.csv` (Test data)
   - `sample.csv` (Sample data for confusion matrix)

3. **Run the script**:

   ```bash
   python classfication-annotations.py
   ```

4. **Outputs**:
   - `predicted_results.csv`: Predicted labels for the test data.
   - `confusion_matrix_sample.csv`: Confusion matrix for sample data.

## License

MIT License.


# Binary-Classification-with-Neural-Network
This repository contains code for training a binary classification model using a neural network. The model is primarily designed to identify additional suitable Z-scheme heterojunctions, alongside their corresponding labels.

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
   - `hetero_lattice_Allen_010_diff_06_20221205-label.csv` (Training data)

3. **Run the script**:

   ```bash
   python classfication-annotations.py
   ```

4. **Outputs**:
   - `predicted_results.csv`: Predicted labels for the candidate Z-scheme heterostructures.
## License

MIT License.


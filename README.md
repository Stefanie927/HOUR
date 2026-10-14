# HOUR

This repository is an implementation of the paper 'Learning high-order user-item relation via hyperedge for recommender system'

### 1. Dataset Preparation

We use the Beidian dataset from the [SHGCN](https://github.com/ziruizhu/SHGCN/tree/master/data/Beidian) project. Please download the dataset and place it in the `data/` directory.

### 2. Construct Dataset

The following steps use the Beidian dataset as an example.

(1) **Identify relevant users and items**
   
   Run the following command to perform matrix factorization and identify the most relevant user IDs and item IDs for each hyperedge:
   ```bash
   python construct_data/MF.py
   ```
(2) **Construct the dataset**
   
   Run the following command to construct the datasets for training and testing:
   ```bash
   python construct_data/construct_data.py
   ```

### 3. Training

Run the following command to train the model:

```bash
python run_main.py
```

### 4. Best Model

The best model trained on the Beidian dataset is available in [Releases](https://github.com/Stefanie927/HOUR/releases/tag/v1.0).

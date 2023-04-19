# A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction

This codebase is the implementation of the [GNN-RNN](https://arxiv.org/pdf/2111.08900.pdf) model for crop yield prediction in the US. GNN-RNN is the first machine learning method that embeds geographical knowledge in crop yield prediction and predicts crop yields at the county level nationwide.

## Requirements
- Python 3
- PyTorch 1.0+

Older versions might work as well.

## Data

Sample crop yield dataset is stored in `data/` and the nation-wise adjacency map is stored in `map/`.

## Experiment scripts

For all methods, make sure to check the `test_year`, `model`, and `crop_type` parameters. If `train_week` and `validation_week` are set to 52, no masking is performed.

**Basic regression:**

`baseline/simple_run_train.sh` contains basic methods (linear regression, gradient boosting regressor, MLP). It's slow though.

**Single-year models:**

From the "baselines" directory, run 

`./single_year_run_train.sh cnn`

`./single_year_run_train.sh gru`

`./single_year_run_train.sh lstm`

**CNN-RNN and RNN (5-year models):**

From the "cnn-rnn" directory, run

`./run_train.sh cnn_rnn`

`./run_train.sh rnn`

# A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction

<div align=center><img src="figs/gnn_rnn.png" width="85%"></div>

This codebase is the implementation of the [GNN-RNN](https://arxiv.org/pdf/2111.08900.pdf) model for crop yield prediction in the US (AAAI 2022). GNN-RNN is the first machine learning method that embeds geographical knowledge in crop yield prediction and predicts crop yields at the county level nationwide.

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


## Paper

If you find our work inspiring, please consider citing the following paper:

```bibtex
@inproceedings{fan2022gnn,
  title={A GNN-RNN approach for harnessing geospatial and temporal information: application to crop yield prediction},
  author={Fan, Joshua and Bai, Junwen and Li, Zhiyun and Ortiz-Bobea, Ariel and Gomes, Carla P},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={36},
  number={11},
  pages={11873--11881},
  year={2022}
}
```

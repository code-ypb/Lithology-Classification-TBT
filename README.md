# TCN-BiLSTM-Transformer for Well-Log Lithology Identification

A hybrid deep learning framework combining **Temporal Convolutional Networks (TCN)**, **Bidirectional Long Short-Term Memory (BiLSTM)**, and **Transformer Encoder** for lithology identification from well-log data in sandstone-type uranium deposits.

## Architecture Overview

```
Input (Window) → TCN + SE Attention → BiLSTM → Positional Encoding
    → Transformer Encoder → Multi-Scale Head → LayerNorm → FC → Lithology Class
```

Key innovations:
- **TCN with SE attention**: Captures multi-scale temporal patterns with channel-wise attention
- **BiLSTM**: Models bidirectional sequential dependencies
- **Transformer Encoder**: Captures long-range self-attention with pre-norm and GELU activation
- **Multi-Scale Feature Head**: Gated combination of linear and non-linear feature extraction
- **Class-Balanced Focal Loss**: Handles severe class imbalance via effective number weighting

## Project Structure

```
TCN-BiLSTM-Transformer-Lithology/
├── src/
│   ├── core/                          # Core model components
│   │   ├── model.py                   # TCN-BiLSTM-Transformer model definition
│   │   ├── loss.py                    # Class-Balanced Focal Loss
│   │   ├── feature_engineering.py     # Well-log feature engineering
│   │   ├── postprocess.py             # CRF-like and ensemble post-processing
│   │   └── augmentation.py            # Oversampling, Mixup, weighted sampling
│   │
│   ├── baselines/                     # Baseline comparison models
│   │   ├── cnn.py                     # 1D-CNN baseline
│   │   ├── lstm.py                    # LSTM baseline
│   │   ├── rnn.py                     # Vanilla RNN baseline
│   │   └── ml_baselines.py            # SVM and XGBoost baselines
│   │
│   ├── ablation/                      # Ablation study
│   │   └── ablation_study.py          # Structural ablation experiments
│   │
│   └── utils/                         # Utilities
│       ├── data_loader.py             # Data loading and preprocessing
│       ├── metrics.py                 # Evaluation metrics
│       └── visualization.py           # Plotting functions
│
├── scripts/
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Model evaluation script
│   └── predict.py                     # Single-well prediction script
│
├── generate_sample_data.py            # Synthetic data generator
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/TCN-BiLSTM-Transformer-Lithology.git
cd TCN-BiLSTM-Transformer-Lithology
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data

```bash
python generate_sample_data.py --output_dir sample_data/ --num_wells 9
```

### 2. Train the Model

```bash
python scripts/train.py --data_dir sample_data/ --output_dir output/ --n_epochs 50
```

For hyperparameter optimization with Optuna:

```bash
python scripts/train.py --data_dir sample_data/ --output_dir output/ --n_trials 30 --n_epochs 100
```

### 3. Evaluate

```bash
python scripts/evaluate.py --checkpoint output/best_model.pt --data_dir sample_data/ --output_dir eval_output/
```

### 4. Predict on a Single Well

```bash
python scripts/predict.py --input sample_data/Well_01.csv --checkpoint output/best_model.pt --output predictions.csv
```

## Data Format

The expected CSV format for well-log data:

| Depth | CAL | DEN | DT | GR | RT | SP | Facies |
|-------|-----|-----|----|----|----|----|--------|
| 100.0 | 249.8 | 2.55 | 250.3 | 45.2 | 80.5 | -65.3 | 0 |
| 100.125 | 249.9 | 2.56 | 252.1 | 48.7 | 78.2 | -64.8 | 0 |

Required columns:
- `Depth`: Depth measurement (m)
- `CAL`: Caliper log
- `DEN`: Density log (g/cm³)
- `DT`: Sonic travel time (µs/ft)
- `GR`: Gamma ray log (API)
- `RT`: Resistivity log (Ω·m)
- `SP`: Spontaneous potential (mV)
- `Facies`: Integer lithology label (0–4)

Facies encoding:

| Label | Lithology |
|-------|-----------|
| 0 | Medium Sandstone |
| 1 | Mudstone |
| 2 | Glutenite |
| 3 | Siltstone |
| 4 | Coarse Sandstone |

## Model Hyperparameters

Optimized via Optuna with the following search space:

| Parameter | Search Range | Description |
|-----------|-------------|-------------|
| `tcn_channels` | [32,64], [32,64,128], [64,128] | TCN channel sizes per block |
| `lstm_hidden` | [64, 128, 256] | BiLSTM hidden dimension |
| `lstm_layers` | [1, 2] | BiLSTM depth |
| `nhead` | [2, 4, 8] | Transformer attention heads |
| `trans_fwd` | [64, 128, 256] | Transformer feedforward dimension |
| `trans_layers` | [1, 3] | Transformer encoder layers |
| `dropout` | [0.1, 0.5] | Universal dropout rate |
| `lr` | [1e-4, 1e-2] (log) | Learning rate |
| `batch_size` | [256, 512, 1024] | Mini-batch size |

## Post-Processing

Three post-processing methods are applied to raw predictions:

1. **Median Filter**: Applies `scipy.ndimage.median_filter` with kernel_size=3
2. **CRF-like Smoothing**: Combines local voting with probability confidence
3. **Adaptive Median Filter**: Applies median filter only to low-confidence predictions
4. **Ensemble**: Majority vote among the three methods above

## Interpretability

The framework includes multiple interpretability methods:

- Integrated Gradients
- Self-Attention Visualization
- Saliency Maps
- Temporal Importance
- SHAP

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Author2025,
  title={TCN-BiLSTM-Transformer for Lithology Identification from Well-Log Data in Sandstone-Type Uranium Deposits},
  author={Author},
  journal={Journal of Geophysical Research: Solid Earth},
  year={2025},
  publisher={AGU}
}
```

## Data Availability

The anonymized well-log dataset used in this study is deposited at Zenodo with restricted access (doi:10.5281/zenodo.XXXXXXX). Access can be requested for academic research purposes by contacting the corresponding author. Well identifiers have been anonymized (Well1–Well8, TestWell) to protect sensitive geological location information. A synthetic data generator (`generate_sample_data.py`) is included in this repository for testing and demonstration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

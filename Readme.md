# TASR-ReImp
Transformer-based Affective State Recognition Reimplementation

This project implements a Transformer-based architecture for multi-modal emotion recognition using physiological signals (heart rate, galvanic skin response, and motion/accelerometer data) from the DAPPER dataset.

## Overview

The model consists of three main components:

1. **Feature Extraction and Embedding**: CNN-based extractors process each physiological signal separately, followed by a cross-attention mechanism to fuse information across modalities.

2. **Transformer Encoder**: Self-attention layers capture temporal and cross-modal dependencies, enabling the model to learn complex patterns in the data.

3. **Classification Head**: A feed-forward network classifies the encoded features into emotional states.

## Model Architecture

The framework follows this process:
- Extract features from each modality using CNNs
- Fuse modalities with parallel cross-attention
- Concatenate and add positional encoding
- Process through Transformer encoder
- Classify with a feed-forward network

## Classification Tasks

The model supports three classification tasks:

1. **Binary Classification (PANAS-based)**: 
   - Class 0 (Negative): Sum of negative affect PANAS items (indices 0, 1, 2, 3, 5, 8)
   - Class 1 (Positive): Sum of positive affect PANAS items (indices 4, 6, 7, 9)
   - Classification is determined by comparing which sum is higher

2. **5-Class Classification (Valence-based)**: 
   - Classes are directly mapped from valence scores (1-5) to class indices (0-4):
     - Valence 1 → Class 0
     - Valence 2 → Class 1
     - Valence 3 → Class 2
     - Valence 4 → Class 3
     - Valence 5 → Class 4

3. **5-Class Classification (Arousal-based)**: 
   - Classes are directly mapped from arousal scores (1-5) to class indices (0-4):
     - Arousal 1 → Class 0
     - Arousal 2 → Class 1
     - Arousal 3 → Class 2
     - Arousal 4 → Class 3
     - Arousal 5 → Class 4

## DAPPER Dataset Preprocessing

The preprocessing includes two main scripts:
1. `denoise.py` - Uses adaptive LMS filtering and median filtering to denoise physiological data from the DAPPER dataset
2. `align.py` - Aligns denoised physiological data with ESM events, extracting 30 minutes of data before each ESM event

### Dataset Acquisition

The DAPPER dataset can be accessed via the following link:
https://doi.org/10.7303/syn22418021

### Data Structure - aligned_events.pkl

The `align.py` script generates an `aligned_events.pkl` file with the following structure:

```
aligned_events = {
    participant_id (int): {  # Participant ID, e.g., 1001
        "event1": {  # First ESM event
            "heart_rate": np.ndarray(1800,),  # Heart rate data, 1Hz sampling, 1800 points for 30 minutes
            "motion": np.ndarray(1800,),      # Motion intensity data, 1Hz sampling, 1800 points for 30 minutes
            "GSR": np.ndarray(1800,),         # Galvanic skin response data, 1Hz sampling, 1800 points for 30 minutes
            "panas": np.ndarray(10,),         # PANAS emotional rating data, 10 items
            "valence": int,                    # Valence rating (1-5)
            "arousal": int,                    # Arousal rating (1-5)
            "end_time": str                    # Event end time (i.e., ESM questionnaire start time)
        },
        "event2": { ... },   # Second ESM event
        ...
    },
    participant_id (int): { ... },  # Next participant
    ...
}
```

### Data Descriptions

1. **Physiological Data**:
   - `heart_rate`: Heart rate data in bpm (beats per minute), 1Hz sampling rate
   - `motion`: Root mean square of triaxial acceleration, representing overall motion intensity, unit is m/s², 1Hz sampling rate
   - `GSR`: Galvanic skin response data in μS (microsiemens), 1Hz sampling rate

2. **Emotional Data**:
   - `panas`: Positive and Negative Affect Schedule ratings for 10 items, each scored 1-5
   - `valence`: Emotional valence rating, range 1-5, higher scores indicate more positive emotions
   - `arousal`: Emotional arousal rating, range 1-5, higher scores indicate higher arousal

3. **Time Information**:
   - `end_time`: Start time of the ESM questionnaire, format 'YYYY/MM/DD HH:MM:SS'
   - The extracted physiological data is for the 30 minutes before this time point

### Data Processing Features

- Missing data is handled using forward fill (ffill) and backward fill (bfill) methods
- Only events with complete 1800 seconds (30 minutes) of data are kept
- Data is sorted and aligned by time
- All physiological data has undergone adaptive noise cancellation (LMS algorithm) and median filtering

### PANAS Item Descriptions

- PANAS_1: upset (心烦意乱的)
- PANAS_2: hostile (充满敌意的)
- PANAS_3: alert (警觉的)
- PANAS_4: ashamed (惭愧的)
- PANAS_5: inspired (受鼓舞的)
- PANAS_6: nervous (紧张的)
- PANAS_7: determined (坚决的)
- PANAS_8: attentive (专注的)
- PANAS_9: afraid (害怕的)
- PANAS_10: active (积极活跃的)

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Required packages: numpy, matplotlib, sklearn, tqdm

Install dependencies:

```bash
pip install torch numpy matplotlib scikit-learn tqdm
```

### Running the Model

To train the model with default parameters:

```bash
python pipeline.py --data_path path/to/aligned_events.pkl
```

To run with specific hyperparameters:

```bash
python pipeline.py --data_path path/to/aligned_events.pkl --task va5 --batch_size 64 --epochs 100 --learning_rate 1e-3
```

For arousal-based classification:

```bash
python pipeline.py --data_path path/to/aligned_events.pkl --task ar5 --batch_size 64 --epochs 100
```

### Batch Experiments

The `run_experiments.sh` script is provided to run multiple experiments with different hyperparameter configurations:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This script will:
1. Run experiments across different categories (default, model architecture, optimization, regularization, batch size)
2. Log all results and create a summary report
3. Generate visualizations for each experiment

### Hyperparameter Tuning

The pipeline supports various command-line arguments for tuning:

#### Model Architecture

- `--feature_dim`: Feature dimension for each modality (default: 512)
- `--use_projection`: Whether to use projection layer (flag)
- `--projection_dim`: Dimension after projection (default: None)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of Transformer layers (default: 4)
- `--d_ff`: Feed-forward dimension in Transformer (default: 2048)
- `--classifier_hidden_dim`: Classifier hidden dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.2)

#### Task Settings

- `--task`: Classification task, either 'binary', 'va5' (valence-based), or 'ar5' (arousal-based) (default: 'va5')

#### Training Settings

- `--train_ratio`: Train/test split ratio (default: 0.8)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay coefficient (default: 1e-5)
- `--beta1`: Beta1 for Adam optimizer (default: 0.9)
- `--beta2`: Beta2 for Adam optimizer (default: 0.999)
- `--epsilon`: Epsilon for Adam optimizer (default: 1e-8)
- `--patience`: Early stopping patience (default: 10)
- `--gpu`: GPU ID to use, -1 for CPU (default: 0)

#### Data Settings

- `--data_path`: Path to the aligned_events.pkl file (default: 'aligned_events.pkl')
- `--seq_len`: Sequence length (default: 1800)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--seed`: Random seed (default: 42)

#### Output Settings

- `--checkpoint_dir`: Directory for saving checkpoints (default: 'checkpoints')

## Evaluation Metrics

The model is evaluated using:
- Accuracy: Proportion of correct predictions
- Precision: Proportion of true positives among predicted positives
- Macro F1 Score: Harmonic mean of precision and recall, averaged across classes

## Results

Trained models, logs, and plots will be saved in the checkpoint directory.

Each run generates:
- Model checkpoints for best F1 scores
- Training history plots
- Results pickle file with complete metrics

## Implementation Details

- **Feature Extraction**: CNN with layers of size 128 and 512
- **Normalization**: LayerNorm instead of BatchNorm to avoid cross-subject information leakage
- **Fusion**: Parallel cross-attention mechanism where each modality attends to other modalities
- **Optimization**: Adam optimizer with linear learning rate decay
- **Regularization**: Dropout of 0.2 and L2 regularization with coefficient 1e-5

## Training and Evaluation

The model is trained using the following procedure:

1. **Data Splitting**: The DAPPER dataset is split by subjects into three sets:
   - 70% of subjects for training
   - 10% of subjects for validation (used for early stopping and hyperparameter tuning)
   - 20% of subjects for testing (used only for final evaluation)
   
   This subject-based splitting ensures no data leakage between sets.

2. **Training**: The model is trained using the Adam optimizer with the following parameters:
   - Learning rate: 1e-3 with linear decay
   - Batch size: 64
   - Betas: (0.9, 0.999)
   - Epsilon: 1e-8
   - Weight decay (L2 regularization): 1e-5
   - Dropout rate: 0.2
   - Early stopping with patience of 10 epochs

3. **Evaluation**: The model is evaluated on the test set using accuracy, precision, and F1 score.

## File Structure

- `feature_extraction.py`: Feature extractors and multi-modal embedding
- `encoder.py`: Transformer encoder implementation
- `classifier.py`: Classification head and metrics calculation
- `pipeline.py`: Complete training and evaluation pipeline
- `run_all_experiments.sh`: Script to run multiple experiments with different configurations

## Reference

This is an independent reimplementation of the model described in the research paper:

```
@article{sensors25030761,
  title={Transformer-Driven Affective State Recognition from Wearable Physiological Data in Everyday Contexts},
  url={https://doi.org/10.3390/s25030761},
  DOI={10.3390/s25030761}
}
```

Results may differ from the original work due to implementation details.
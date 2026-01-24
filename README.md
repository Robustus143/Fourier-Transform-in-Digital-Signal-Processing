# Musical Instrument Classification Using Digital Signal Processing and Machine Learning

## Overview

This project implements a comprehensive system for **automatic musical instrument classification** using Fourier Transform analysis, spectral features, and machine learning algorithms. The system analyzes audio signals from six orchestral instruments (saxophone, flute, viola, cello, trumpet, oboe) and classifies them using advanced signal processing techniques and multiple machine learning approaches.

## Project Objectives

The primary goal of this study was to investigate whether **traditional spectral analysis methods combined with machine learning** can achieve reliable classification of musical instruments. The project explores:

- Audio signal feature extraction using librosa
- Mel-Frequency Cepstral Coefficients (MFCC) analysis
- Spectral feature analysis (centroid, bandwidth, roll-off, etc.)
- Deep learning approaches using Artificial Neural Networks (ANN)
- Support Vector Machines (SVM) for classification
- Comparative analysis of different classification algorithms

## Key Features

### Audio Features Extracted
1. **Chroma Short-Time Fourier Transform (Chroma STFT)** - Chromatic distribution of energy
2. **RMSE (Root Mean Square Energy)** - Signal energy level
3. **Spectral Centroid** - "Brightness" of the sound
4. **Spectral Bandwidth** - Width of the spectral distribution
5. **Spectral Roll-off** - Frequency below which 85% of energy is concentrated
6. **Zero Crossing Rate** - Number of sign changes in the signal
7. **Mel-Frequency Cepstral Coefficients (MFCC)** - 13-20 coefficients capturing perceptual characteristics

### Instruments Classified
- **Saxophone (sax)**
- **Flute**
- **Viola**
- **Cello**
- **Trumpet**
- **Oboe**

## Project Structure

```
├── ANN1.ipynb                          # Artificial Neural Network implementation (basic)
├── ANN2.ipynb                          # Artificial Neural Network (advanced)
├── SVM.ipynb                           # Support Vector Machine classifier
├── SVM_modified.ipynb                  # Modified SVM implementation
├── london_phill_dataset_multi.rar      # Audio dataset (London Philharmonic Orchestra)
├── MCOS-DSMMPH_Kuz.pptx               # Presentation on Digital Signal Processing
├── DSMSI_Kuz.pptx                     # Presentation on Signal Processing
├── Дипломна_Кузь.pdf                  # Diploma thesis (Ukrainian)
└── README.md                           # Original documentation
```

## Technologies & Libraries

### Audio Processing
- **librosa** - Audio analysis and feature extraction
- **scipy** - Signal processing tools
- **numpy** - Numerical computations

### Machine Learning & Data Processing
- **scikit-learn** - Machine learning algorithms (SVM, preprocessing)
- **keras/TensorFlow** - Deep learning and neural networks
- **pandas** - Data manipulation and analysis

### Visualization & Analysis
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **PIL** - Image processing for spectrograms

## Methodology

### 1. **Data Preparation**
- Load audio files from London Philharmonic Orchestra dataset (mono, 5-30 second clips)
- Extract audio features using librosa

### 2. **Feature Engineering**
- Compute 27+ features per audio sample:
  - Chroma STFT (mean)
  - RMSE (mean)
  - Spectral features (centroid, bandwidth, roll-off)
  - Zero crossing rate
  - MFCC (13-20 coefficients, averaged)
- Normalize features using StandardScaler

### 3. **Classification Approaches**

#### Artificial Neural Networks (ANN)
- Sequential neural network models
- Multiple hidden layers with dropout regularization
- Adam optimizer
- Categorical cross-entropy loss
- Implementation in ANN1.ipynb and ANN2.ipynb

#### Support Vector Machines (SVM)
- Linear and RBF kernel implementations
- Hyperparameter tuning
- Cross-validation for model evaluation
- Implementation in SVM.ipynb and SVM_modified.ipynb

### 4. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for detailed analysis
- Classification reports
- Performance comparison across algorithms

## Dataset

The project uses the **London Philharmonic Orchestra dataset** (`london_phill_dataset_multi`), containing:
- High-quality recordings of orchestral instruments
- Multiple recordings per instrument
- Various playing techniques and dynamics
- Suitable for training robust classification models

## Implementation Details

### Audio Feature Extraction Parameters
```python
fs = 44100          # Sampling frequency (Hz)
n_fft = 2048        # FFT window size
hop_length = 512    # Samples between successive frames
n_mels = 128        # Mel frequency bands
n_mfcc = 13         # Number of MFCC coefficients
```

### Data Processing Pipeline
1. Load audio files (mono, duration: 5-30 seconds)
2. Compute audio features
3. Create CSV dataset with features and labels
4. Generate spectrograms (Mel-scale) for visualization
5. Train/test split (typically 80/20 or 70/30)
6. Feature scaling (StandardScaler)
7. Train classification models
8. Evaluate and compare results

## Results

The project demonstrates:
- Successful classification of musical instruments from audio signals
- Comparative analysis of ANN vs SVM approaches
- Identification of most discriminative features for each instrument class
- Visualization of spectrograms and classification metrics

## Usage

### Prerequisites
```bash
pip install librosa pandas numpy matplotlib scikit-learn keras tensorflow seaborn pillow
```

### Running the Analysis

1. **Feature Extraction (ANN1.ipynb)**
   - Generates spectrograms from audio files
   - Extracts audio features into CSV format
   - Prepares dataset for machine learning

2. **ANN Classification (ANN1.ipynb or ANN2.ipynb)**
   - Load prepared dataset
   - Train neural network model
   - Evaluate on test set
   - Visualize results

3. **SVM Classification (SVM.ipynb or SVM_modified.ipynb)**
   - Load prepared dataset
   - Train SVM classifier with various kernels
   - Cross-validation and hyperparameter tuning
   - Compare with ANN results
   - Generate confusion matrices and metrics

## Key Findings

- **Spectral features** (MFCC, spectral centroid) are highly discriminative for instrument classification
- **Combined approach** using multiple features outperforms single-feature classification
- **Deep learning (ANN)** and **traditional ML (SVM)** both achieve high accuracy rates
- **Instrument-specific characteristics** are captured effectively by the feature extraction pipeline

## References & Documentation

- **MCOS-DSMMPH_Kuz.pptx** - Comprehensive presentation on Digital Signal Processing and Musical Instrument Classification
- **DSMSI_Kuz.pptx** - Digital Signal Processing concepts and methods
- **Дипломна_Кузь.pdf** - Complete diploma thesis with detailed methodology and results (in Ukrainian)

## Technical Contributions

This project demonstrates:
1. **Signal Processing**: Practical application of FFT, spectrograms, and MFCC
2. **Feature Engineering**: Extraction of meaningful audio features
3. **Machine Learning**: Implementation and comparison of classification algorithms
4. **Data Science Workflow**: Complete pipeline from raw audio to classification results

## Author

Developed as part of a diploma thesis project on Digital Signal Processing and Musical Instrument Analysis.

## License

Please refer to the original repository for license information.

## Acknowledgments

- Dataset: London Philharmonic Orchestra
- Signal Processing: librosa library documentation
- Machine Learning: scikit-learn and Keras/TensorFlow documentation

---

## How to Cite

If you use this project in your research, please cite:
- Original repository: Fourier-Transform-in-Digital-Signal-Processing
- Related thesis: Дипломна_Кузь.pdf

---

**Status**: Educational/Research Project | **Year**: 2024-2025 | **Language**: Python/Jupyter Notebooks

# GROUP21\_CENG313 — BirdCLEF 2025: Bird Species Identification

|  |  |
| :---- | :---- |
| **Member** | Mahmut Eren Sarıcaoğlu |
| **Student ID** | 23118080029 |
| **GitHub** | mahmuterensrc |
| **Group ID** | 21 |
| **Kaggle competition** | [BirdCLEF 2025](https://www.kaggle.com/c/birdclef-2025) |

## 1\. The competition

The data consists of continuous, unconstrained environmental soundscapes (audio recordings) from diverse geographical locations. The goal is to build a machine learning model capable of identifying the specific bird species vocalizing in the audio.

* **Input:** `.ogg` audio files categorized by species in `train_audio/` (e.g., `train_audio/amekes/XC5018.ogg`).  
* **Labels:** `train_metadata.csv` → `primary_label` (target species) and `secondary_labels` (background birds).  
* **Metric:** Macro-Averaged F1-Score (to account for severe class imbalances).  
* **Submission:** A CSV file predicting the bird species for given audio segments.

## 2\. Approach

### Assignment 1 — EDA & baseline (`assignment1_eda_baseline.ipynb`)

* **EDA:** Comprehensive structural and acoustic analyses were conducted. Key findings include:  
  * **Class Imbalance:** 28,564 samples across 206 species with a severe long-tailed distribution.  
  * **Duration Variance:** Audio files range from 0.87 to 443 seconds. File counts are deceptive; minority classes have as little as 11 seconds of total audio.  
  * **Metadata Bias:** High presence of `0.0` ratings and "Unknown" recordists, which were retained to allow the model to generalize to noisy, real-world conditions.  
  * **Class Noise:** Frequent occurrence of `secondary_labels`, meaning random crops might capture background birds instead of the primary target.  
* **Preprocessing:** Geolocation features (`latitude`, `longitude`) were dropped. Variable-length 1D audio waveforms were dynamically cropped or padded to a uniform 5-second length and transformed into 2D Mel-spectrograms (128 mel bands) directly on the GPU.  
* **Baseline** (80/20 Stratified Split on Train):  
  * A pre-trained **ResNet-18** architecture, modified to accept single-channel Mel-spectrogram inputs.  
  * Optimization via Adam optimizer and Cross-Entropy Loss.

## 4\. Repository layout

&nbsp;

```
GROUP21_CENG313/
├── README.md                          
├── requirements.txt                   
├── .gitignore                         
├── assignment1_eda_baseline.ipynb             
├── src/
│   ├── __init__.py                    
│   ├── data.py      # PyTorch Dataset, dataloaders, padding/cropping
│   ├── models.py    # ResNet-18 model architecture and builder
│   └── engine.py    # Training & evaluation loop functions
├── data/            # NOT committed (Kaggle rules + file size)
└── submissions/     # Screenshots and submission logs
```

&nbsp;

## 5\. How to run

1. Join the competition on Kaggle and accept the rules.  
2. Go to Kaggle → Settings → API → **Create New Token** to download your `kaggle.json`.  
3. Open `assignment1_eda_baseline.ipynb` in Google Colab (File → Open notebook → GitHub → paste this repo's URL).  
4. Upload your `kaggle.json` file to the root `/content/` directory in Colab.  
5. **Runtime → Run all**. The first cell will automatically clone the repository to the local SSD, download the dataset into `data/`, and extract it.  
6. The notebook will execute the EDA, feature extraction, and train the baseline model.

To run locally instead: `pip install -r requirements.txt`, place the Kaggle dataset files under `data/`, and run the notebook. `data/` is excluded from version control (see `.gitignore`).

## 6\. Progress log

| Date | Work |
| :---- | :---- |
| 01/11/2026 | Repository structure created, GitHub organization transfer completed. |
| 02/11/2026 | PyTorch custom dataset, data loader, and on-the-fly GPU Mel-spectrogram pipeline implemented. |
| 05/11/2026 | Comprehensive EDA (class distribution, exact durations, background noise, metadata quality) was performed. |
| 10/11/2026 | ResNet-18 baseline model trained (10 epochs) and evaluated on a held-out split. |
| 13/11/2026 | Structural and textwise corrections and final reviews have been made. |

## 7\. Kaggle submission history

| Date | Model | Public Score | Private Score |
| :---- | :---- | :---- | :---- |
| 2026-11-12 | ResNet-18 Baseline | 0.6335 (Val Split) | N/A |

Competition closed — evaluated on a held-out split. The highest-scoring submissions on the Kaggle leaderboard achieved \~0.93. The naive baseline is approximately 30% behind these top solutions.
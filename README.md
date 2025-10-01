# [ICCV 2025] VALLR: Visual ASR Language Model for Lip Reading


<img width="1210" height="506" alt="image" src="https://github.com/user-attachments/assets/938343fb-9ede-4c0c-9f12-db417c044e3a" />

[Marshall Thomas](https://www.surrey.ac.uk/people/marshall-thomas ), [Edward Fish](https://ed-fish.github.io/), [Richard Bowden](https://www.surrey.ac.uk/people/richard-bowden)



[![arXiv](https://img.shields.io/badge/arXiv-2503.21408-b31b1b.svg)](https://arxiv.org/abs/2503.21408) 
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

VALLR is a novel two-stage, phoneme-centric framework for **Visual Automatic Speech Recognition (VASR)** that achieves state-of-the-art performance in lip reading. This approach significantly reduces **Word Error Rate (WER)** by first predicting a sequence of phonemes from visual inputs and then using a fine-tuned **Large Language Model (LLM)** to reconstruct coherent words and sentences. This repository contains the official PyTorch implementation of the VALLR model, along with tools for data preprocessing and inference.

---

### Key Features

* **State-of-the-Art Performance**: Achieves a SOTA WER of **18.7% on the LRS3 dataset**, outperforming existing methods.
* **Two-Stage Framework**: Decouples the visual feature extraction from linguistic modeling, leading to improved accuracy and data efficiency.
* **Phoneme-Centric Approach**: By predicting phonemes as an intermediate representation, VALLR effectively handles the ambiguities of visemes and coarticulation effects.
* **Data Efficient**: Requires **99.4% less labeled data** than the next best approach, making it highly practical for real-world applications without the need for self-supervised pre-training.
* **Modular Design**: The codebase is organized into distinct components for data processing, model architecture, and inference pipelines, allowing for easy customization and extension.

---

### Model Architecture

The VALLR model consists of two main components:

1.  **Video-to-Phoneme Network**: A Video Transformer with a CTC head that takes video frames of a speaker's mouth as input and predicts a sequence of phonemes.
2.  **Phoneme-to-Sentence LLM**: A fine-tuned Large Language Model (LLM) that takes the phoneme sequence as input and reconstructs the corresponding words and sentences.

This two-stage design allows the model to first learn the complex visual features of speech and then leverage the linguistic knowledge of an LLM to generate coherent text.

---

### Results

Here's a comparison of VALLR's performance against other state-of-the-art methods on the LRS3 and LRS2 datasets. Our method achieves SOTA performance on LRS3 using only the supervised training set, without any self-supervised pre-training.

#### LRS3 Dataset Comparison

| Method | Unlabeled (hrs) | Labeled (hrs) | WER (%) |
| :--- | :---: | :---: | :---: |
| **Self-supervised pre-training + Supervised fine-tuning** | | | |
| AV-HuBERT Large [44] | 1,759 | 30 | 32.5 |
| Lip2Vec [12] | 1,759 | 30 | 31.2 |
| Whisper [41] | 1,759 | 30 | 25.5 |
| RAVEn [18] | 1759 | 433 | 23.1 |
| USR [19] | 1,326 | 433 | 21.5 |
| **Supervised fine-tuning only** | | | |
| **Ours** | **-** | **30** | **18.7** |

#### LRS2 Dataset Comparison

| Method | Unlabeled (hrs) | Labeled (hrs) | WER (%) |
| :--- | :---: | :---: | :---: |
| **Self-supervised pre-training + Supervised fine-tuning** | | | |
| Sub-Word [40] | 2,676 | 2,676 | 22.6 |
| RAVEn [18] | 1,759 | 223 | 17.9 |
| USR [19] | 1,759 | 223 | 15.4 |
| **Supervised fine-tuning only** | | | |
| **Ours** | **-** | **28** | **20.8** |

---
---

### Getting Started

#### Prerequisites

* Python 3.10 or higher
* PyTorch 2.4.1
* Other dependencies listed in `requirements.txt`

#### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/MarshallT-99/VALLR.git](https://github.com/MarshallT-99/VALLR.git)
    cd VALLR
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Inference

To run inference on a single video, use the `infer` mode and provide the path to the trained model and the video file.

1.  **Download the pretrained model weights**:
    * [Download VALLR Model Weights from Google Drive](https://drive.google.com/file/d/14u7MRTxXL1psHMnlssw_Drq-zRMGC7zD/view?usp=sharing)

2.  **Run inference**:
    ```bash
    python main.py --mode infer --model_path /path/to/your/downloaded/model.pth --infer_video_path /path/to/your/video.mp4
    ```
3.  **Train mode**:
    ```bash
    python3 main.py --mode train --version V1 --save_model_path path/to/model --videos_root path/to/videos
    ```

---

### Codebase Overview

* **`main.py`**: The main script for running inference.
* **`Models/ML_VALLR.py`**: Contains the implementation of the VALLR model.
* **`Data/dataset.py`**: The `VideoDataset` class for loading and preprocessing video data.
* **`face_cropper.py`**: A utility for detecting and cropping faces from video frames using MediaPipe.
* **`config.py`**: Configuration file for setting hyperparameters and other settings.

---


### Citation

If you use this code or the VALLR model in your research, please cite the following paper:

```
@article{thomas2025vallr,
  title={VALLR: Visual ASR Language Model for Lip Reading},
  author={Thomas, Marshall and Fish, Edward and Bowden, Richard},
  journal={arXiv preprint arXiv:2503.21408},
  year={2025}
}
```

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

# NYCU Computer Vision 2026 HW1

* **Student ID:** 111550017
* **Name:** 藍逸薰

## Introduction

I leveraged a  ResNet-101 architecture pre-trained on the ImageNet-1K dataset. This allows the model to utilize sophisticated, pre-learned visual features (edges, textures, and shapes) and then I fine-tune them to adapt this specific 100-class problem.

---

## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
```
---

## Usage

# Training

folder format:
folder
---
  data
    ---
    train
      100 classfolders
    test
      test images
    val
      100 classfolders
  model.py

It will train the model then do one testing when it s done. You will have to download best_model.pth from https://drive.google.com/file/d/1W5Uhz5gB_GPxXubUMzlxlO4LAEdoTtzU/view?usp=sharing or e3 first because my Git large file just doesn't seems to be working. The testing process will have to ensure that it have the training data class folder in data folder in order to get the correct label. 

```bash
python model.py
```
# Testing only

Modify the epoch variable in model.py to 0, then

```bash
python model.py
```
---

## Performance Snapshot

<img width="1182" height="52" alt="Image" src="https://github.com/user-attachments/assets/6e1f3e82-48ad-47b5-ab6c-02d03c91ee6b" />

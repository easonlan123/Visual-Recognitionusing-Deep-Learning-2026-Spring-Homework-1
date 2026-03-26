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

It will train the model then do one testing when it s done.

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

# Automatic Segmentation of the Aortic Wall from CT Scans with U-Net
NOTE: this is a fork of the original repository by [s194255](https://github.com/s194255) and I.

Repository for segmentation of the aortic walls from CT scans using sparse annotations.
![Hej](figures/images/annotering.png) 


# Getting started
Clone this repo:
```bash
git clone https://github.com/TECH-yufu/DTUWallSegmentation.git
cd DTUWallSegmentation
```
Create conda environment and install dependencies:
```bash
conda env create -f requirements.yaml
conda activate DTUAorta
```

# Training
The basic model can be trained with the following
```bash
python train.py --batch_size 4 --img_size 512 --lr 3e-3 --data_path "data_path" --dataloader_type "3D" --contextual 
```

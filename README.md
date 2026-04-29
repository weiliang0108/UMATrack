# UMATrack

This is a PyTorch implementation of A Novel Lightweight Siamese-like Tracker Based on Transformers with Unidirectional Mixed Attention.

## Environment ​Dependencies

- Python 3.9
- Pytorch 2.4.0
- cuda 12.4
- opencv
- pandas
- timm
- PyYAML
- cython
- tqdm
- jpeg4py
- tensorboard

## Data Preparation

We use the following public datasets. Please download them from the official websites:

- TrackingNet: [SilvioGiancola/TrackingNet-devkit: Development kit for TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit)
- GOT-10k: [GOT-10k: Generic Object Tracking Benchmark](http://got-10k.aitestunion.com/)
- COCO: [COCO - Common Objects in Context](https://cocodataset.org/)
- LaSOT: [LaSOT - Large-scale Single Object Tracking](http://vision.cs.stonybrook.edu/~lasot/)

Your dataset directory should follow this structure:

```
-- datasets
		-- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- train2017
     	-- lasot
            |-- airplane
				|-- airplane-1
				|-- airplane-2
				...
            |-- basketball
            |-- bear
            ...
```

# Unsupervised domain adaptation

## Source only
* GTA5
* SYNTHIA

```
python train_sourceonly.py
python evaluate.py
python compute_iou.py
```

## Target supervised learning
* Cityscapes, IDD, Mapillary
* Crosscity (Rio, Rome, Taipei, Tokyo)

```
python train_supervised.py
python evaluate.py
python compute_iou.py
```

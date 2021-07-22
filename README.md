# Race Classifier

## Requirements

* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.8.5 installation. I recommend Anaconda3.
* I recommend PyTorch 1.6.0, but any version will be fine.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 11.1 toolkit and cuDNN 8.x

## Preparing datasets

I used [Align&Cropped Images](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and labeled 10,000 images on by own into 5 group (Caucasian, Black, Asian, Middle Easterners/Indian, Not determined)

I split dataset into train : valid : test = 70 : 15 : 15

```
.
├── data
│   ├── test
│   │   ├── image
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   ├── 000003.jpg
│   │   │   ├── ...
│   │   └── label.csv
│   ├── train
│   │   ├── image
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   ├── 000003.jpg
│   │   │   ├── ...
│   │   └── label.csv
│   └── valid
│       ├── image
│       │   ├── 000001.jpg
│       │   ├── 000002.jpg
│       │   ├── 000003.jpg
│       │   ├── ...
│       └── label.csv
├── eval.py
├── load_data.py
├── __pycache__
│   └── load_data.cpython-38.pyc
├── resnet18_raceRecog_epoch100.pt
└── train.py
```

## Training networks

```
python train.py --n_epochs 300 --model_pth model_300epochs.pt --batch_size 16
```

## Evaluation

```
python eval.py --model_pth model_300epochs.pt
```

## Reference

I wrote README.md with reference to [this page](https://github.com/ji-in/stylegan2)

I referred to a [PyTorch official site](https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html) for fine codes

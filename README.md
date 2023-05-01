# Lip to Speech Synthesis in the Wild with Multi-task Learning

This repository contains the PyTorch implementation of the following paper:
> **Lip to Speech Synthesis in the Wild with Multi-task Learning**<br>
> Minsu Kim, Joanna Hong, and Yong Man Ro<br>
> \[[Paper](https://arxiv.org/abs/2302.08841)\] \[[Demo Video](https://github.com/joannahong/Lip-to-Speech-Synthesis-in-the-Wild)\]

<div align="center"><img width="30%" src="img/Img.PNG?raw=true" /></div>

## Requirements
- python 3.7
- pytorch 1.6 ~ 1.8
- torchvision
- torchaudio
- sentencepiece
- ffmpeg
- av
- tensorboard
- scikit-image 0.17.0 ~
- opencv-python 3.4 ~
- pillow
- librosa
- pystoi
- pesq
- scipy
- einops
- ctcdecode

### Datasets
#### Download
LRS2/LRS3 dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/

For data preprocessing, download the lip coordinate of LRS2 and LRS3 from the below links. 
- [LRS2](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EYr_pFzvluxJumdVbQ7c6iwB0Va7rRheS-NIZigZMejOyQ?e=WYsPvX)
- [LRS3](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EW05i6UFVVlPlpdyWEmSvtMBEelCSh6Wm2jJTw4MgIGwgQ?e=ReY3k4)

Unzip and put the files to
```
./data/LRS2/LRS2_crop/*.txt
./data/LRS3/LRS3_crop/*.txt
```

#### Preprocessing
After download the dataset, extract audio file (.wav) from the video.
We suppose the data directory is constructed as
```
LRS2-BBC
├── main
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt

LRS2-BBC_audio
├── main
|   ├── *
|   |   └── *.wav
```

```
LRS3-TED
├── trainval
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt

LRS2-TED_audio
├── trainval
|   ├── *
|   |   └── *.wav
```

Moreover, put the train/val/test splits to <br>
```
./data/LRS2/*.txt
./data/LRS3/*.txt
```

For the LRS2, we use the original splits of the dataset provided. [LRS2](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EYr_pFzvluxJumdVbQ7c6iwB0Va7rRheS-NIZigZMejOyQ?e=WYsPvX) <br>
For the LRS3, we use the unseen splits setting of [SVTS](https://arxiv.org/abs/2205.02058), where they are placed in the directory already.

## Training the Model
`data_name` argument is used to choose which dataset will be used. (LRS2 or LRS3) <br>
To train the model, run following command:

```shell
# Data Parallel training example using 4 GPUs on LRS2
python train_LRS.py \
--data '/data_dir_as_like/LRS2-BBC' \
--data_name 'LRS2'
--checkpoint_dir 'enter_the_path_to_save' \
--batch_size 80 \
--epochs 200 \
--dataparallel \
--gpu 0,1,2,3
```

```shell
# 1 GPU training example on LRS3
python train_LRS.py \
--data '/data_dir_as_like/LRS3-TED' \
--data_name 'LRS3'
--checkpoint_dir 'enter_the_path_to_save' \
--batch_size 80 \
--epochs 200 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--augmentations`: whether performing augmentation
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--window_size`: number of frames to be used for training
- Refer to `train_LRS3.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy time costs of waveform conversion (griffin-lim). <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation metrics, generated mel-spectrogram, and audio


## Testing the Model
To test the model, run following command:
```shell
# test example on LRS2
python test_LRS.py \
--data 'data_directory_path' \
--data_name 'LRS2'
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 20 \
--save_mel \
--save_wav \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--save_mel`: whether to save the 'mel_spectrogram' and 'spectrogram' in `.npz` format
- `--save_wav`: whether to save the 'waveform' in `.wav` format
- Refer to `test.py` for the other parameters


## Pre-trained model checkpoints
We provide pre-trained VCA-GAN models trained on LRS2 and LRS3. <br>
The performances are reported in our ICASSP23 [paper](https://arxiv.org/abs/2302.08841). 

|       Dataset       |   STOI   |
|:-------------------:|:--------:|
|LRS2 |   [0.407](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EU6QAhptiR9Ns-WRVghuHYwBA_22Wp4EMr0O3LWZkQjnpw?e=qwkAu9)  |
|LRS3 |   [0.474](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EXnSqRowtL5PiOyk1UL2qK4BgmCvRF4WAZhmi5v6qY0RlA?e=XlBFGC)  |


## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2023lip,
  title={Lip-to-Speech Synthesis in the Wild with Multi-task Learning},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  journal={arXiv preprint arXiv:2302.08841},
  year={2023}
}
```

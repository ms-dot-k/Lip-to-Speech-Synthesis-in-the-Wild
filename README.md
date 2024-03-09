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
- [LRS2](https://drive.google.com/file/d/10cnzNRRr-LQbS5kjc393FLvmNxPJ_u1N/view?usp=sharing)
- [LRS3](https://drive.google.com/file/d/10eAVKBuy7TyslcPdv4xmf5dxSYx4NMrS/view?usp=sharing)

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

For the LRS2, we use the original splits of the dataset provided. <br>
For the LRS3, we use the unseen splits setting of [SVTS](https://arxiv.org/abs/2205.02058), where they are placed in the directory already.

## Pretrained Visual Frontend
Training to read sentences using CTC loss is hard to find optimization points. <br>
We provide the visual frontend pre-trained on LRS2 and LRS3 using CTC. <br>
When you training from scratch, it is good to initialize the visual frontend with the checkpoints below.

- [LRS2](https://drive.google.com/file/d/1-I_v_RA_A73hCGqZjSP2nYLeLiR0SP7X/view?usp=sharing) <br>
- [LRS3](https://drive.google.com/file/d/1-jq2vPc3_znejLsuFEq2PueN5saqtkN3/view?usp=sharing)

## Training the Model
`data_name` argument is used to choose which dataset will be used. (LRS2 or LRS3) <br>
To train the model, run following command:

```shell
# Data Parallel training example using 2 GPUs on LRS2
python train.py \
--data '/data_dir_as_like/LRS2-BBC' \
--data_name 'LRS2'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 16 \
--epochs 200 \
--eval_step 3000 \
--dataparallel \
--gpu 0,1
```

```shell
# 1 GPU training example on LRS3
python train.py \
--data '/data_dir_as_like/LRS3-TED' \
--data_name 'LRS3'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 8 \
--epochs 200 \
--eval_step 3000 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--asr_checkpoint` : pretrained ASR checkpoint
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--output_content_on`: when the output content supervision is turned on (reconstruction loss)
- Refer to `train.py` for the other training parameters

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
python test.py \
--data 'data_directory_path' \
--data_name 'LRS2'
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 20 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- Refer to `test.py` for the other parameters


## Pre-trained model checkpoints
The pre-trained ASR models for output-level content supervision and lip-to-speech synthesis models on LRS2 and LRS3 are available. <br>

| Model |       Dataset       |   STOI   |
|:-------------------:|:-------------------:|:--------:|
|ASR|LRS2 |   [Link](https://drive.google.com/file/d/1-A1FIy8tLumY0otQQzX-MHoG5SM7E9u5/view?usp=sharing)  |
|ASR|LRS3 |   [Link](https://drive.google.com/file/d/1-ba1I2S9vI5v5R6uQXo4E8f3rTuDS9S0/view?usp=sharing)  |
|Lip2Speech|LRS2 |   [0.526](https://drive.google.com/file/d/1-IPQ5DF3iwkeVRlbNLR58G8rZJ6kwe26/view?usp=sharing)  |
|Lip2Speech|LRS3 |   [0.497](https://drive.google.com/file/d/1-hSBVopSc7gqELPvWbYll_xh-0ESKFoP/view?usp=sharing)  |


## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{kim2023lip,
  title={Lip-to-speech synthesis in the wild with multi-task learning},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

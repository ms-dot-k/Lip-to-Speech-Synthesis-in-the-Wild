import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import Crop, StatefulRandomHorizontalFlip
from PIL import Image
import librosa
from matplotlib import pyplot as plt
import glob
from scipy import signal
import torchvision
import cv2
from torch.autograd import Variable
from librosa.filters import mel as librosa_mel_fn
from src.data.audio_processing import dynamic_range_compression, dynamic_range_decompression, griffin_lim
from src.data.stft import STFT
import sentencepiece as spm
import math
log1e5 = math.log(1e-5)

class MultiDataset(Dataset):
    def __init__(self, data, mode, max_v_timesteps=155, min_window_size=30, max_window_size=50, max_text_length=150, augmentations=False, num_mel_bins=80, fast_validate=False):
        assert mode in ['train', 'test', 'val']
        self.data = data
        self.mode = mode
        self.sample_window = True if (mode == 'train') else False
        self.fast_validate = fast_validate
        self.max_v_timesteps = max_v_timesteps
        self.max_text_len = max_text_length
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.augmentations = augmentations if (mode == 'train') else False
        self.num_mel_bins = num_mel_bins
        self.file_paths, self.file_names, self.crops = self.build_file_list(data, mode)
        self.f_min = 55.
        self.f_max = 7500.
        self.stft = TacotronSTFT(filter_length=1024, hop_length=160, win_length=640, n_mel_channels=num_mel_bins, sampling_rate=16000, mel_fmin=self.f_min, mel_fmax=self.f_max)

        self.info = {}
        self.info['video_fps'] = 25
        self.info['audio_fps'] = 16000
        self.sp = spm.SentencePieceProcessor(model_file='./data/lrs2lrs3_lower.model')
        self.char_list = []
        with open('./data/lrs2lrs3_lower.vocab', encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                self.char_list.append(l.strip().split("\t")[0])
        self.num_characters = self.sp.get_piece_size()

    def build_file_list(self, lrs3, mode):
        file_list, paths = [], []
        crops = {}

        ## LRS3 crop (lip centered axis) load ##
        file = open(f"./data/LRS3/LRS3_crop/preprocess_pretrain.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['pretrain/' + file] = crop_str
        file = open(f"./data/LRS3/LRS3_crop/preprocess_test.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['test/' + file] = crop_str
        file = open(f"./data/LRS3/LRS3_crop/preprocess_trainval.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['trainval/' + file] = crop_str

        ## LRS3 file lists##
        file = open(f"./data/LRS3/lrs3_unseen_{mode}.txt", "r")
        content = file.read()
        file.close()
        for file in content.splitlines():
            if file in crops:
                file_list.append(file)
                paths.append(f"{lrs3}/{file}")

        print(f'Mode: {mode}, File Num: {len(file_list)}')
        return paths, file_list, crops

    def build_tensor(self, frames, crops):
        if self.augmentations:
            s = random.randint(-5, 5)
        else:
            s = 0
        crop = []
        for i in range(0, len(crops), 2):
            left = int(crops[i]) - 50 + s
            upper = int(crops[i + 1]) - 50 + s
            right = int(crops[i]) + 50 + s
            bottom = int(crops[i + 1]) + 50 + s
            crop.append([left, upper, right, bottom])
        crops = crop

        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])

        temporalVolume = torch.zeros(frames.size(0), 1, 112, 112)
        for i, frame in enumerate(frames):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                Crop(crops[i]),
                transforms.Resize([112, 112]),
                augmentations1,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(0.4136, 0.1700),
            ])
            temporalVolume[i] = transform(frame)

        ### Random Erasing ###
        if self.augmentations:
            #### spatial erasing
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :, np.maximum(0, y_s):np.minimum(112, y_s + 56), np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_names[idx]
        file_path = self.file_paths[idx]
        start_sec = 0
        stop_sec = None

        content = open(file_path + ".txt", "r").read()
        crops = self.crops[file].split("/")
        if 'pretrain' in file_path:
            content, start_sec, stop_sec = self.get_pretrain_words(content)
        else:
            content = content.splitlines()[0][7:].strip().lower()

        cap = cv2.VideoCapture(file_path + '.mp4')
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        cap.release()
        audio, _ = librosa.load(file_path.replace('LRS3-TED', 'LRS3-TED_audio') + '.wav', sr=16000)
        vid = torch.tensor(np.stack(frames, 0))
        audio = torch.tensor(audio).unsqueeze(0)

        if not 'video_fps' in self.info:
            self.info['video_fps'] = 25
            self.info['audio_fps'] = 16000

        if vid.size(0) < 5 or audio.size(1) < 5:
            vid = torch.zeros([1, 112, 112, 3])
            audio = torch.zeros([1, int(3 * 16000 / 25)])

        if 'pretrain' in file_path:
            st_v_frame = math.floor(start_sec * self.info['video_fps'])
            end_v_frame = math.ceil(stop_sec * self.info['video_fps'])
            st_a_frame = math.floor(start_sec * self.info['audio_fps'])
            end_a_frame = math.ceil(stop_sec * self.info['audio_fps'])

            vid = vid[st_v_frame:end_v_frame]
            audio = audio[:, st_a_frame:end_a_frame]
        else:
            st_v_frame = 0

        ## Video ##
        vid = vid.permute(0, 3, 1, 2)  # T C H W
        num_v_frames = vid.size(0)
        if num_v_frames > self.max_v_timesteps:
            print(f"Cutting Video frames off. Requires {num_v_frames} frames: {file}. But Max_timestep is {self.max_v_timesteps}")
            vid = vid[:self.max_v_timesteps]
            num_v_frames = vid.size(0)
            num_a_frame = round(num_v_frames / self.info['video_fps'] * self.info['audio_fps'])
            audio = audio[:, :num_a_frame]

        crops = crops[st_v_frame * 2:st_v_frame * 2 + num_v_frames * 2]

        ## Audio ##
        aud = audio / torch.abs(audio).max() * 0.9
        aud = torch.FloatTensor(self.preemphasize(aud.squeeze(0))).unsqueeze(0)
        aud = torch.clamp(aud, min=-1, max=1)

        melspec, spec = self.stft.mel_spectrogram(aud)

        melspec = self.normalize(melspec)

        spec = self.normalize_spec(spec)    # 0 ~ 1
        spec = self.stft.spectral_normalize(spec)   # log(1e-5) ~ 0 # in log scale
        spec = self.normalize(spec)   # -1 ~ 1

        index = random.randint(0, max(0, melspec.size(2) - 50))
        sp_mel = melspec[:, :, index:index + 50]

        if self.sample_window:
            melspec, spec, audio, start_frame, window_size, num_a_frames = self.extract_window(vid, melspec, spec, audio, self.info)
        else:
            start_frame = 0
            window_size = vid.size(0)
            num_a_frames = melspec.size(2)
            melspec = nn.ConstantPad2d((0, window_size * 4 - melspec.size(2), 0, 0), 0.)(melspec)
            spec = nn.ConstantPad2d((0,  window_size * 4 - spec.size(2), 0, 0), 0.)(spec)

        vid = self.build_tensor(vid, crops)

        audio_length = int(window_size / self.info['video_fps'] * self.info['audio_fps'])
        audio = audio[:, :audio_length]

        target, target_len = self.encode(content)

        return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames, audio_length, target, target_len, start_frame, window_size, file_path.replace(self.data, '')[1:], sp_mel

    def get_pretrain_words(self, content):
        lines = content.splitlines()[4:]
        words = []
        for line in lines:
            word, start, stop, _ = line.split(" ")
            start, stop = float(start), float(stop)
            words.append([word, start, stop])

        num_words = min(random.randint(4, 20), len(words))   #4 ~ 20
        word_start = random.randint(0, len(words) - num_words)
        word_end = word_start + num_words

        sample_start = 0
        sample_end = 0
        content = ""
        for word in words[word_start:word_end]:
            word, start, end = word
            if sample_start == 0:
                sample_start = start
            if end > sample_end:
                sample_end = end
            content = content + " " + word

        return content.lower(), sample_start, sample_end

    def encode(self, content):
        encoded = self.sp.encode(content)
        if len(encoded) > self.max_text_len:
            print(f"Max output length too short. Required {len(encoded)}. But Max text_length is {self.max_text_len}")
            encoded = encoded[:self.max_text_len]
        num_txt = len(encoded)
        return encoded, num_txt

    def extract_window(self, vid, mel, spec, aud, info):
        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / 160

        window_size = min(random.randint(self.min_window_size, self.max_window_size), vid.size(0))

        st_fr = random.randint(0, vid.size(0) - window_size)

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(window_size * vid_2_aud)

        mel = mel[:, :, st_mel_fr:st_mel_fr + mel_window_size]
        spec = spec[:, :, st_mel_fr:st_mel_fr + mel_window_size]

        if mel.size(2) < mel_window_size:
            mel = nn.ConstantPad2d((0, mel_window_size - mel.size(2), 0, 0), 0.)(mel)
            spec = nn.ConstantPad2d((0, mel_window_size - spec.size(2), 0, 0), 0.)(spec)

        num_a_frames = mel.size(2)

        aud = aud[:, st_mel_fr*160:st_mel_fr*160 + mel_window_size*160]
        aud = torch.cat([aud, torch.zeros([1, int(window_size / info['video_fps'] * info['audio_fps'] - aud.size(1))])], 1)
        return mel, spec, aud, st_fr, window_size, num_a_frames

    def collate_fn(self, batch):
        vid_lengths, spec_lengths, padded_spec_lengths, aud_lengths, target_lens = [], [], [], [], []
        for data in batch:
            vid_lengths.append(data[3])
            spec_lengths.append(data[5])
            aud_lengths.append(data[6])
            padded_spec_lengths.append(data[0].size(2))
            target_lens.append(data[8])

        max_vid_length = max(vid_lengths)
        max_aud_length = max(aud_lengths)
        max_spec_length = max(padded_spec_lengths)
        max_target_length = max(target_lens)

        padded_vid = torch.zeros(len(batch), 1, max_vid_length, 112, 112)
        padded_targets = []
        padded_melspec = []
        padded_spec = []
        padded_audio = []
        start_frames = []
        window_sizes = []
        f_names = []
        sp_mels = []

        for i, (melspec, spec, vid, num_v_frames, audio, spec_len, audio_length, target, target_len, start_frame, window_size, f_name, sp_mel) in enumerate(batch):
            padded_vid[i, :, :num_v_frames, :, :] = vid   # B, C, T, H, W
            padded_targets.append(target + [0] * (max_target_length - target_len))
            padded_melspec.append(nn.ConstantPad2d((0, max_spec_length - melspec.size(2), 0, 0), 0.0)(melspec))
            padded_spec.append(nn.ConstantPad2d((0, max_spec_length - spec.size(2), 0, 0), 0.0)(spec))
            sp_mels.append(nn.ConstantPad2d((0, 100 - sp_mel.size(2), 0, 0), 0.)(sp_mel))
            padded_audio.append(torch.cat([audio, torch.zeros([max_aud_length - audio.size(0)])], 0))
            start_frames.append(start_frame)
            window_sizes.append(window_size)
            f_names.append(f_name)

        vid = padded_vid.float()
        vid_length = torch.IntTensor(vid_lengths)
        targets = torch.LongTensor(padded_targets)
        targets_length = torch.IntTensor(target_lens)
        start_frame = torch.LongTensor(start_frames)
        window_size = torch.LongTensor(window_sizes)
        melspec = torch.stack(padded_melspec, 0).float()
        sp_mels = torch.stack(sp_mels, 0).float()
        spec = torch.stack(padded_spec, 0).float()
        spec_length = torch.IntTensor(spec_lengths)
        audio = torch.stack(padded_audio, 0).float()

        return melspec, spec, vid, vid_length, audio, spec_length, targets, targets_length, start_frame, window_size, f_names, sp_mels

    def arr2txt(self, arr):
        return self.sp.decode(arr.numpy().tolist())

    def inverse_mel(self, mels, mel_len, stft):
        wavs = []
        if len(mels.size()) < 4:
            mels = mels.unsqueeze(0)    # B,1,80,T
        for kk, mel in enumerate(mels):
            mel = mel.unsqueeze(0)            # 1,1,80,T
            mel = mel[:, :, :, :mel_len[kk]]
            mel = self.denormalize(mel)
            mel = stft.spectral_de_normalize(mel)
            mel = mel.transpose(2, 3).contiguous()  # B,80,T --> B,T,80
            spec_from_mel_scaling = 1000
            spec_from_mel = torch.matmul(mel, stft.mel_basis)
            spec_from_mel = spec_from_mel.transpose(2, 3).squeeze(1)  # B,1,F,T
            spec_from_mel = spec_from_mel * spec_from_mel_scaling

            wav = griffin_lim(spec_from_mel, stft.stft_fn, 60).squeeze(1).squeeze(0)  # L
            wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
            wav = self.deemphasize(wav)
            wav = np.clip(wav, -1, 1)
            wavs.append(wav)
        return wavs

    def inverse_spec(self, specs, mel_len, stft):
        wavs = []
        if len(specs.size()) < 4:
            specs = specs.unsqueeze(0)  # B,1,321,T
        for kk, spec in enumerate(specs):
            spec = spec.unsqueeze(0)            # 1,1,321,T
            spec = spec[:, :, :, :mel_len[kk]]
            spec = self.denormalize(spec)  # log1e5 ~ 0
            spec = stft.spectral_de_normalize(spec)  # 0 ~ 1
            spec = self.denormalize_spec(spec)  # 0 ~ 14
            wav = griffin_lim(spec.squeeze(1), stft.stft_fn, 60).squeeze(1).squeeze(0)  # L
            wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
            wav = self.deemphasize(wav)
            wav = np.clip(wav, -1, 1)
            wavs.append(wav)
        return wavs

    def preemphasize(self, aud):
        aud = signal.lfilter([1, -0.97], [1], aud)
        return aud

    def deemphasize(self, aud):
        aud = signal.lfilter([1], [1, -0.97], aud)
        return aud

    def normalize(self, melspec):
        melspec = ((melspec - log1e5) / (-log1e5 / 2)) - 1    #0~2 --> -1~1
        return melspec

    def denormalize(self, melspec):
        melspec = ((melspec + 1) * (-log1e5 / 2)) + log1e5
        return melspec

    def normalize_spec(self, spec):
        spec = (spec - spec.min()) / (spec.max() - spec.min())  # 0 ~ 1
        return spec

    def denormalize_spec(self, spec):
        spec = spec * 14. # 0 ~ 14
        return spec

    def audio_preprocessing(self, aud):
        fc = self.f_min
        w = fc / (16000 / 2)
        b, a = signal.butter(7, w, 'high')
        aud = aud.squeeze(0).numpy()
        aud = signal.filtfilt(b, a, aud)
        return torch.tensor(aud.copy()).unsqueeze(0)

    def plot_spectrogram_to_numpy(self, mels):
        fig, ax = plt.subplots(figsize=(15, 4))
        im = ax.imshow(np.squeeze(mels, 0), aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = self.save_figure_to_numpy(fig)
        plt.close()
        return data

    def save_figure_to_numpy(self, fig):
        # save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data.transpose(2, 0, 1)

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output, magnitudes
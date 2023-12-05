import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.model import Visual_front, Conformer_encoder, CTC_classifier, Speaker_embed, Mel_classifier
from src.models.asr_model import ASR_model
# from ctcdecode import CTCBeamDecoder
import editdistance
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.vid_aud_lrs2 import MultiDataset as LRS2_Dataset
from src.data.vid_aud_lrs3 import MultiDataset as LRS3_Dataset
from src.data.vid_aud_grid import MultiDataset as grid_Dataset
from torch.nn import DataParallel as DP
import torch.nn.parallel
import time
import glob
from torch.autograd import grad
from pesq import pesq
from pystoi import stoi
from matplotlib import pyplot as plt
import copy
import librosa
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="/work/lixiaolou/data/grid/video/")
    parser.add_argument('--data_name', default="grid", help='LRS2, LRS3, grid')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/')
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/Best_0110_stoi_0.671_estoi_0.448_pesq_nan.ckpt')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=3000)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--augmentations", default=True)
    parser.add_argument("--mask_prob", type=float, default=0.5)

    parser.add_argument("--min_window_size", type=int, default=50)
    parser.add_argument("--max_window_size", type=int, default=50)
    parser.add_argument("--mode", type=str, default='test', help='train, test, val')
    parser.add_argument("--max_timesteps", type=int, default=250)

    parser.add_argument("--conf_layer", type=int, default=12)
    parser.add_argument("--num_head", type=int, default=8)

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    v_front = Visual_front(in_channels=1, conf_layer=args.conf_layer, num_head=args.num_head)
    mel_layer = Mel_classifier()
    sp_layer = Speaker_embed()

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        mel_layer.load_state_dict(checkpoint['mel_layer_state_dict'])
        sp_layer.load_state_dict(checkpoint['sp_layer_state_dict'])
        del checkpoint

    v_front.cuda()
    mel_layer.cuda()
    sp_layer.cuda()

    if args.dataparallel:
        v_front = DP(v_front)
        mel_layer = DP(mel_layer)
        sp_layer = DP(sp_layer)

    _ = test(v_front, mel_layer, sp_layer, fast_validate=False)

def test(v_front, mel_layer, sp_layer, fast_validate=False):
    with torch.no_grad():
        v_front.eval()
        mel_layer.eval()
        sp_layer.eval()

        if args.data_name == 'LRS2':
            val_data = LRS2_Dataset(
                data=args.data,
                mode=args.mode,
                min_window_size=args.min_window_size,
                max_window_size=args.max_window_size,
                max_v_timesteps=args.max_timesteps,
                augmentations=args.augmentations,
            )
        elif args.data_name == 'LRS3':
            val_data = LRS3_Dataset(
                data=args.data,
                mode=args.mode,
                min_window_size=args.min_window_size,
                max_window_size=args.max_window_size,
                max_v_timesteps=args.max_timesteps,
                augmentations=args.augmentations,
            )
        elif args.data_name == 'grid':
            val_data = grid_Dataset(
            data=args.data,
            mode=args.mode,
            min_window_size=args.min_window_size,
            max_window_size=args.max_window_size,
            max_v_timesteps=args.max_timesteps,
            augmentations=args.augmentations,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=True if fast_validate else False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            # collate_fn=lambda x: val_data.collate_fn(x),
            collate_fn = val_data.collate_fn,
        )

        stft = copy.deepcopy(val_data.stft).cuda()
        criterion = nn.L1Loss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(10 * batch_size, int(len(dataloader.dataset)))
            max_batches = 10
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        stoi_list = []
        estoi_list = []
        pesq_list = []

        description = 'Check test step' if fast_validate else 'Test'
        print(description)
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            mel, spec, vid, vid_len, wav_tr, mel_len, targets, target_len, _, _, f_name, _ = batch

            sp_feat = sp_layer(mel[:, :, :, :50].cuda())
            v_feat = v_front(vid.cuda(), vid_len.cuda())  # S,B,512

            g_mel = mel_layer(v_feat, sp_feat)

            loss = criterion(g_mel, mel.cuda()).cpu().item()
            val_loss.append(loss)

            wav_pred = val_data.inverse_mel(g_mel, mel_len, stft)
            for _ in range(g_mel.size(0)):
                min_len = min(len(wav_pred[_]), len(wav_tr[_]))
                stoi_list.append(stoi(wav_tr[_][:min_len], wav_pred[_][:min_len], 16000, extended=False))
                estoi_list.append(stoi(wav_tr[_][:min_len], wav_pred[_][:min_len], 16000, extended=True))
                try:
                    pesq_list.append(pesq(8000, librosa.resample(wav_tr[_][:min_len].numpy(), 16000, 8000), librosa.resample(wav_pred[_][:min_len], 16000, 8000), 'nb'))
                except:
                    continue

                m_name, v_name, file_name = f_name[_].split('/')
                if not os.path.exists(f'./generate/test_{args.data_name}/mel/{m_name}/{v_name}'):
                    print('0')
                    os.makedirs(f'./generate/test_{args.data_name}/mel/{m_name}/{v_name}')
                np.savez(f'./generate/test_{args.data_name}/mel/{m_name}/{v_name}/{file_name}.npz',
                         mel=g_mel[_, :, :, :mel_len[_]].detach().cpu().numpy())

                if not os.path.exists(f'./generate/test_{args.data_name}/wav/{m_name}/{v_name}'):
                    os.makedirs(f'./generate/test_{args.data_name}/wav/{m_name}/{v_name}')
                sf.write(f'./generate/test_{args.data_name}/wav/{m_name}/{v_name}/{file_name}.wav', wav_pred[_], 16000, subtype='PCM_16')

            if i >= max_batches:
                break

        print('val_stoi:', np.mean(np.array(stoi_list)))
        print('val_estoi:', np.mean(np.array(estoi_list)))
        # print('val_pesq:', np.mean(np.array(pesq_list)))
        with open(f'./test_{args.data_name}.txt', 'w') as f:
            f.write(f'STOI : {np.mean(stoi_list)}\n')
            f.write(f'ESTOI : {np.mean(estoi_list)}\n')
            # f.write(f'PESQ : {np.mean(pesq_list)}\n')
        # return np.mean(np.array(val_loss)), np.mean(np.array(stoi_list)), np.mean(np.array(estoi_list)), np.mean(np.array(pesq_list))
        return np.mean(np.array(val_loss)), np.mean(np.array(stoi_list)), np.mean(np.array(estoi_list))


def wer(predict, truth):
    word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
    wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
    return wer


if __name__ == "__main__":
    args = parse_args()
    train_net(args)


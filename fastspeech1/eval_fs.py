import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
from torch.utils.data import DataLoader
import text
from fastspeech1 import model_fs as M
from dataset.dataset_fs import BufferDataset
from dataset.dataset_fs import get_data_to_buffer, collate_fn_tensor
from fastspeech1.loss_fs import DNNLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    test1 = "I am very happy to see you again!"
    test2 = "Durian model is a very good speech synthesis!"
    test3 = "When I was twenty, I fell in love with a girl."
    test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
    test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
    test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
    data_list = list()
    data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test4, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test5, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test6, hp.text_cleaners))
    return data_list


def evaluate(model, step, vocoder=None):
    torch.manual_seed(0)

    # Get dataset
    print("Load data to buffer")
    buffer = get_data_to_buffer('train.txt')
    dataset = BufferDataset(buffer)

    # Get Training Loader
    validating_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)

    # Get Loss
    fastspeech_loss = DNNLoss().to(device)

    d_l = []
    mel_l = []
    mel_p_l = []

    for i, batchs in enumerate(validating_loader):
        # real batch start here
        for j, db in enumerate(batchs):
            start_time = time.perf_counter()
            # Get Data
            character = db["text"].long().to(device)
            mel_target = db["mel_target"].float().to(device)
            duration = db["duration"].int().to(device)
            mel_pos = db["mel_pos"].long().to(device)
            src_pos = db["src_pos"].long().to(device)
            max_mel_len = db["mel_max_len"]

            with torch.no_grad():
                # Forward
                mel_output, mel_postnet_output, duration_predictor_output = model(character,
                                                                                  src_pos,
                                                                                  mel_pos=mel_pos,
                                                                                  mel_max_length=max_mel_len,
                                                                                  length_target=duration)

                # Cal Loss
                mel_loss, mel_postnet_loss, duration_loss = fastspeech_loss(mel_output,
                                                                            mel_postnet_output,
                                                                            duration_predictor_output,
                                                                            mel_target,
                                                                            duration)
                total_loss = mel_loss + mel_postnet_loss + duration_loss

                d_l.append(duration_loss.item())
                mel_l.append(mel_loss.item())
                mel_p_l.append(mel_postnet_loss.item())


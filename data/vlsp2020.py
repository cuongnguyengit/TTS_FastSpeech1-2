import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
from text import _clean_text
import hparams as hp
from collections import defaultdict


def prepare_align(in_dir):
    for r, d, f in os.walk(in_dir):
        for file in f:
            if file.endswith(".txt"):
                basename = file.replace('.txt', '')
                with open(os.path.join(r, file), 'r', ) as rf:
                    text = rf.read().strip()
                    text = _clean_text(text, hp.text_cleaners)

                with open(os.path.join(in_dir, '{}.txt'.format(basename)), 'w') as f1:
                    f1.write(text)


def build_from_path(in_dir, out_dir):
    print('In dir', in_dir)
    print('Out dir', out_dir)

    dict_tmp = defaultdict(list)
    # r=root, d=directories, f = files
    for r, d, f in os.walk(in_dir):
        for file in f:
            if file.endswith(".txt"):
                basename = file.replace('.txt', '')
                dict_tmp[basename].append('.txt')
            if file.endswith(".wav"):
                basename = file.replace('.wav', '')
                dict_tmp[basename].append('.wav')
    dem = 0
    for r, d, f in os.walk(os.path.join(out_dir, 'TextGrid/')):
        for file in f:
            if file.endswith(".TextGrid"):
                dem += 1
                basename = file.replace('.TextGrid', '')
                dict_tmp[basename].append('.TextGrid')
    print(dem)
    list_unuse = []

    for k, v in dict_tmp.items():
        if len(v) <= 2:
            print(k, v)
            list_unuse.append(k)


    index = 1
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0

    for r, d, f in os.walk(in_dir):
        for file in f:
            # try:
            if file.endswith(".txt"):
                # print(file)
                basename = file.replace('.txt', '')
                if basename in list_unuse:
                    continue
                ret = process_utterance(in_dir, out_dir, basename)
                if ret is None:
                    continue
                else:
                    info, f_max, f_min, e_max, e_min, n = ret

                if basename[:5][2:] in ['001', '002', '003', '004', '005']:
                    val.append(info)
                else:
                    train.append(info)
                if index % 100 == 0:
                    print("Done %d" % index)
                index = index + 1
                f0_max = max(f0_max, f_max)
                f0_min = min(f0_min, f_min)
                energy_max = max(energy_max, e_max)
                energy_min = min(energy_min, e_min)
                n_frames += n
            # except Exception as e:
            #     print(file, e)


    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames * hp.hop_length / hp.sampling_rate / 3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s + '\n')

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename))
    if not os.path.exists(tg_path):
        print(tg_path, ' is not found')
        return None
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{' + '}{'.join([i for i in phone if len(i) >= 1]) + '}'  # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')  # '{A}{B} {C}'
    text = text.replace('}{', ' ')  # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    # print(_)
    wav = wav[int(hp.sampling_rate * start):int(hp.sampling_rate * end)].astype(np.float32)
    # wav = wav[:, 0]
    # print(wav.shape, len(wav))
    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length / hp.sampling_rate * 1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)

    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), \
           mel_spectrogram.shape[1]

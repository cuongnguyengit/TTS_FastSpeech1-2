import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from unidecode import unidecode

from fastspeech2.model_fs2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
from string import punctuation
from G2p import G2p
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def preprocess(text):
    text = text.rstrip(punctuation).lower()
    print(text)
    g2p = G2p(args.dict_path)
    phone = g2p.g2p(text)
    # phone = list(filter(lambda p: p != ' ', phone))
    print(phone)
    phone = '{' + '}{'.join(phone.split()) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(text, hp.text_cleaners))
    print(sequence_to_text(sequence))
    print(sequence)
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)


def get_FastSpeech2(args):
    checkpoint_path = os.path.join(
        args.checkpoint_path, "checkpoint_{}.pth.tar".format(args.step))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, waveglow, text, sentence, prefix='', duration_control=1.0, pitch_control=1.0, energy_control=1.0):
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control)

    # mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    # mel = mel[0].cpu().transpose(0, 1).detach()
    # mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    # f0_output = f0_output[0].detach().cpu().numpy()
    # energy_output = energy_output[0].detach().cpu().numpy()

    if not os.path.exists(args.test_path):
        os.makedirs(args.test_path)
    name = ''.join(unidecode(sentence).split())
    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
    #     hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, name)))
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(
            args.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, name[:10])))

    # utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], [
    #                 'Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, name)))

def gen_mel(model, text, idx, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control)
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_filename = '{}-mel.npy'.format(idx)
    np.save(os.path.join(args.test_path, mel_filename), mel, allow_pickle=False)
    mel_filename = '{}-mel-post.npy'.format(idx)
    np.save(os.path.join(args.test_path, mel_filename), mel_postnet, allow_pickle=False)


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=300000)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--path', type=str, default='test.txt')
    parser.add_argument('--dict_path', type=str, default='syllable_g2p.txt')
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)

    args = parser.parse_args()
    sentences = []
    with open(args.path, 'r', encoding='utf-8') as rf:
        lines = rf.read().split('\n')
        for i, line in enumerate(lines):
            sentences.append(line)

    model = get_FastSpeech2(args).to(device)
    # waveglow = utils.get_waveglow()
    waveglow = None
    with torch.no_grad():
        for idx, sentence in enumerate(sentences):
            text = preprocess(sentence)
            print(text.shape)
            # synthesize(model, waveglow, text, sentence, 'step_{}'.format(
            #     args.step), args.duration_control, args.pitch_control, args.energy_control)
            gen_mel(model, text, idx, duration_control=1.0, pitch_control=1.0, energy_control=1.0)
            print('DONE', idx)

    # print(text_to_sequence('{o2_T1 l e2_T1 sp k o1_T3 t ie2_T3 ng ng uoi3_T2 sp n oi_T3}', ['basic_cleaners']))

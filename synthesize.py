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
# import audio as Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def preprocess(text):
    # text = text.rstrip(punctuation)
    print('Text', text)
    sequence = np.array(text_to_sequence(text, []))
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


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=300000)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--sentence', type=str, default='xin chào các bạn .')
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)

    args = parser.parse_args()

    sentences = [
        "{o2_T1 l e2_T1 sp k o1_T3 t ie2_T3 ng ng uoi3_T2 sp n oi_T3}",
        # "thế từ chỗ này không nhìn thấy cầu à",
        # 'bệnh viêm phổi lạ khởi phát từ vũ hán , trung quốc , sau được xác định là cô vít mười chín , không chỉ khiến'
        # ' trung quốc phong tỏa hơn sáu mươi triệu dân mà nó cũng nhanh chóng trở thành đại dịch toàn cầu , '
        # 'khiến hơn bảy tư triệu người mắc bệnh , trong đó hơn một phẩy sáu triệu người tử vong'
    ]
    sentences.append(args.sentence)

    model = get_FastSpeech2(args).to(device)
    waveglow = utils.get_waveglow()
    with torch.no_grad():
        for sentence in sentences:
            text = preprocess(sentence)
            print(text.shape)
            synthesize(model, waveglow, text, sentence, 'step_{}'.format(
                args.step), args.duration_control, args.pitch_control, args.energy_control)

    # print(text_to_sequence('{o2_T1 l e2_T1 sp k o1_T3 t ie2_T3 ng ng uoi3_T2 sp n oi_T3}', ['basic_cleaners']))

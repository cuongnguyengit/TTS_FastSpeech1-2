from fastspeech2 import train_fs2
from fastspeech1 import train_fs
import torch
import argparse

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

def run(args):
    if args.model == 'FastSpeech2':
        train_fs2.main(args, device)
    elif args.model == 'FastSpeech1':
        train_fs.main(args, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--model', type=str, default='FastSpeech2')
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()

    run(args)
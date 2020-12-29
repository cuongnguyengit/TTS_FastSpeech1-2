import torch
import argparse
import hparams as hp
device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

def run(args):
    if args.model == 'FastSpeech2':
        from fastspeech2 import train_fs2
        train_fs2.main(args, device)
    elif args.model == 'FastSpeech1':
        from fastspeech1 import train_fs
        train_fs.main(args, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--model', type=str, default='FastSpeech2')
    parser.add_argument('--name_task', type=str, default='FastSpeech2')
    parser.add_argument('--data_path', type=str, default='/content/vlsp2020_01_fs2_gen_audio/')
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    hp.data_path = args.data_path
    hp.name_task = args.name_task
    if not os.path.exists(os.path.join(hp.root_path, hp.name_task)):
        os.makedirs(os.path.join(hp.root_path, hp.model))

    run(args)
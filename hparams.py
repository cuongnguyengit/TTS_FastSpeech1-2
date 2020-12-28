import os

dataset = "vlsp2020"
data_path = '/content/vlsp2020/'
model = 'FastSpeech1'  # 'FastSpeech2'
# Text
text_cleaners = ['basic_cleaners']

root_path = '/content/drive/MyDrive/voice_data/'

checkpoint_path = os.path.join(root_path, model, "ckpt", dataset)
synth_path = os.path.join(root_path, model, "synth", dataset)
eval_path = os.path.join(root_path, model, "eval", dataset)
log_path = os.path.join(root_path, model, "log", dataset)
test_path = os.path.join(root_path, model, 'results')

# Audio and mel
### for VLSP2020 ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

# Quantization for F0 and energy
### for VLSP2020 ###
f0_min = 71.0
f0_max = 741.4
energy_min = 0.0
energy_max = 320.9

if model == 'FastSpeech1':
    # FastSpeech 1
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    batch_size = 32
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32

elif model == 'FastSpeech2':
    # FastSpeech 2
    encoder_layer = 4
    encoder_head = 2
    encoder_hidden = 256
    decoder_layer = 4
    decoder_head = 2
    decoder_hidden = 256
    fft_conv1d_filter_size = 1024
    fft_conv1d_kernel_size = (9, 1)
    encoder_dropout = 0.2
    decoder_dropout = 0.2

    variance_predictor_filter_size = 256
    variance_predictor_kernel_size = 3
    variance_predictor_dropout = 0.5

    max_seq_len = 1000

    # Optimizer
    batch_size = 16
    epochs = 1000
    n_warm_up_step = 4000
    grad_clip_thresh = 1.0
    acc_steps = 1

    betas = (0.9, 0.98)
    eps = 1e-9
    weight_decay = 0.

    # Vocoder
    vocoder = 'waveglow'  # 'waveglow' or 'melgan'

    # Log-scaled duration
    log_offset = 1.

    # Save, log and synthesis
    save_step = 10000
    synth_step = 1000
    eval_step = 1000
    eval_size = 256
    log_step = 1000
    clear_Time = 20
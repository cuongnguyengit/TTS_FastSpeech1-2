import os

dataset = "vlsp2020"
data_path = '/content/vlsp2020/'
name_task = 'FastSpeech2'  # 'FastSpeech2'
waveglow_path = '/content/drive/MyDrive/voice_data/waveglow_78000'
# Text
# text_cleaners = ['basic_cleaners']
text_cleaners = []

root_path = '/content/drive/MyDrive/voice_data/'

checkpoint_path = os.path.join(root_path, name_task, "ckpt", dataset)
synth_path = os.path.join(root_path, name_task, "synth", dataset)
eval_path = os.path.join(root_path, name_task, "eval", dataset)
log_path = os.path.join(root_path, name_task, "log", dataset)
test_path = os.path.join(root_path, name_task, 'results')

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
f0_max = 786.6
energy_min = 0.0
energy_max = 321.4


# Vocoder
vocoder = 'waveglow'  # 'waveglow' or 'melgan'

# Log-scaled duration
log_offset = 1.

# Save, log and synthesis
save_step = 20000
synth_step = 10000
eval_step = 10000
eval_size = 256
log_step = 1000
clear_Time = 20

n_bins = 256

batch_size = 32
epochs = 1000
batch_expand_size = 32
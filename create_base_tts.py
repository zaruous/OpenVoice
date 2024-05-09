import os
import torch
from openvoice.api import BaseSpeakerTTS

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'demo_result'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# Run the base speaker tts
text = "Hi my name is kim young jun"
src_path = f'{output_dir}/base_tts.wav'
base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)


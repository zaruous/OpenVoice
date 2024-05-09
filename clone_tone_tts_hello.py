import os
import torch
from openai import OpenAI
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

## FFMPEG 설치 후 환경변수 세팅 필요
## https://github.com/BtbN/FFmpeg-Builds/releases

ckpt_converter = f'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)


output_dir = "demo_result"

#소리의 글자
base_speaker = f"{output_dir}/openai_source_alloy_output.mp3"
source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)

reference_speaker = 'resources/아이유.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

src_path = 'demo_result/openai_source_output.mp3'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


save_path = f'{output_dir}/output_crosslingual_1.wav'
# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
    message=encode_message)

print(save_path, "complete")
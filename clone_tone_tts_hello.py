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

inputText = "안녕하세요 저는 김영준입니다."


base_speaker = f"{output_dir}/openai_source_output.mp3"
source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)

reference_speaker = 'resources/task-1-02-reference.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)


# Run the base speaker tts
text = [
    "MyShell is a decentralized and comprehensive platform for discovering, creating, and staking AI-native apps.",
    "MyShell es una plataforma descentralizada y completa para descubrir, crear y apostar por aplicaciones nativas de IA.",
    "MyShell est une plateforme décentralisée et complète pour découvrir, créer et miser sur des applications natives d'IA.",
    "MyShell ist eine dezentralisierte und umfassende Plattform zum Entdecken, Erstellen und Staken von KI-nativen Apps.",
    "MyShell è una piattaforma decentralizzata e completa per scoprire, creare e scommettere su app native di intelligenza artificiale.",
    "MyShellは、AIネイティブアプリの発見、作成、およびステーキングのための分散型かつ包括的なプラットフォームです。",
    "MyShell — это децентрализованная и всеобъемлющая платформа для обнаружения, создания и стейкинга AI-ориентированных приложений.",
    "MyShell هي منصة لامركزية وشاملة لاكتشاف وإنشاء ورهان تطبيقات الذكاء الاصطناعي الأصلية.",
    "MyShell是一个去中心化且全面的平台，用于发现、创建和投资AI原生应用程序。",
    "MyShell एक विकेंद्रीकृत और व्यापक मंच है, जो AI-मूल ऐप्स की खोज, सृजन और स्टेकिंग के लिए है।",
    "MyShell é uma plataforma descentralizada e abrangente para descobrir, criar e apostar em aplicativos nativos de IA."
]
text = ["비슷한것같은데 안비슷한가."]

src_path = f'{output_dir}/tmp.wav'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

for i, t in enumerate(text):

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=t,
    )

    response.write_to_file(src_path)

    save_path = f'{output_dir}/output_crosslingual_{i}.wav'

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)
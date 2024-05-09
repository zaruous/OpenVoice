from openai import OpenAI
from dotenv import load_dotenv
import os


print(os.getenv("OPENAI_API_KEY"))
# Please create a file named .env and place your
# OpenAI key as OPENAI_API_KEY=xxx
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
output_dir = "demo_result"


inputText = "This audio will be used to extract the base speaker tone color embedding. " + \
        "Typically a very short audio should be sufficient, but increasing the audio " + \
        "length will also improve the output audio quality."
inputText = "안녕하세요 저는 김영준입니다. 만나서 반갑습니다. OpenVoice를 통한 테스트 음성입니다. " \
            "현재 날짜는 2024년 5월 9일입니다. "

voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
for voice in voices:
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=inputText,

    )
    response.write_to_file(f"{output_dir}/openai_source_{voice}_output.mp3")


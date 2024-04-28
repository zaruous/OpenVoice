from openai import OpenAI
from dotenv import load_dotenv
import os


print(os.getenv("OPENAI_API_KEY"))
# Please create a file named .env and place your
# OpenAI key as OPENAI_API_KEY=xxx
load_dotenv()
os.get
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
output_dir = "demo_result"

inputText = "안녕하세요 저는 김영준입니다."
inputText = "This audio will be used to extract the base speaker tone color embedding. " + \
        "Typically a very short audio should be sufficient, but increasing the audio " + \
        "length will also improve the output audio quality."



response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=inputText
)
response.write_to_file(f"{output_dir}/openai_source_output.mp3")
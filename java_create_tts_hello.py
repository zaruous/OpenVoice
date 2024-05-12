from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

## 외부에서 호출할때 사용

voice = "alloy"
message = "This audio will be used to extract the base speaker tone color embedding. " + \
          "Typically a very short audio should be sufficient, but increasing the audio " + \
          "length will also improve the output audio quality."
output_dir = "demo_result"

if __name__ == "__main__":
    arguments = sys.argv
    print(arguments)
    if len(arguments) >= 2:
        outFileName = arguments[1]
    if len(arguments) >= 3:
        voice = arguments[2]
    if len(arguments) >= 4:
        message = arguments[3]

print(os.getenv("OPENAI_API_KEY"))
# Please create a file named .env and place your
# OpenAI key as OPENAI_API_KEY=xxx
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
inputText = message
response = client.audio.speech.create(
    model="tts-1",
    voice=voice,
    input=inputText,
)
response.write_to_file(f"{output_dir}/" + outFileName)


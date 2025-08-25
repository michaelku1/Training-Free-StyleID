import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/audio/stable-audio-2/inpaint",
    headers={
        "authorization": f"Bearer sk-MYAPIKEY",
        "accept": "audio/*"
    },
    files={
        "audio": open("./sample_data/accordion/accordion1.wav", "rb"),
    },
    data={
        "prompt": "inpaint audio to 8 seconds without changing the content, style, rhythm and other characteristics.",
        "output_format": "mp3",
        "duration": 20,
        "steps": 30,
    },
)

if response.status_code == 200:
    with open("./output.mp3", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))
import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio",
    headers={"authorization": f"Bearer sk-MYAPIKEY", "accept": "audio/*"},
    files={
        "audio": open("/home/mku666/riffusion-hobby/stable_audio_api/sample_data/accordion/accordion1.wav", "rb"),
    },
    data={
        # "prompt": "A song in the 3/4 time signature that features cheerful acoustic guitar, live recorded drums, and rhythmic claps, The mood is happy and up-lifting.",
        "prompt": "style transfer: accordion to violin",
        "output_format": "wav",
        "duration": 5,
        "steps": 30,
        "model": "stable-audio-2.5",
    },
)

if response.status_code == 200:
    with open("./stable_audio_api/output/accordion_to_violin.wav", "wb") as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))
import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio",
    headers={"authorization": f"Bearer sk-MYAPIKEY", "accept": "audio/*"},
    files={"none": ""},
    data={
        "prompt": "A song in the 3/4 time signature that features cheerful acoustic guitar, live recorded drums, and rhythmic claps, The mood is happy and up-lifting.",
        "output_format": "mp3",
        "duration": 20,
        "steps": 30,
        "model": "stable-audio-2.5",
    },
)

if response.status_code == 200:
    with open("./output.mp3", "wb") as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))
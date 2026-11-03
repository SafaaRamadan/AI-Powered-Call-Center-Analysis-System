#call api and get response


import requests
from pathlib import Path
from calls.testing.preprocessing import preprocess_audio 

API_URL = "https://overture-gothic-stability.ngrok-free.dev/analyze"



recording_text_folder = Path("calls/services/records")
records_folder = Path("../Dataset")
preprocessed_folder = Path("../Preprocessed")

def analyze_call_test(audio_path):
    with open(audio_path, "rb") as audio:
        response = requests.post(
            API_URL,
            files={"file": audio}
        )

    response.raise_for_status()
    return response.json()

import os

def preprocessing_stage():
    for file in recording_text_folder.glob("*.txt"):
        file_name = file.stem
        file_path = records_folder / f"{file_name}.wav"
        if not os.path.exists(f"../Preprocessed/{file_name}.wav"):
            final_path = preprocess_audio(file_path, file_name)
            print(final_path)
        # print(recording_audio_folder / f"{file_name}.wav")
        # response = call_api(recording_audio_folder / f"{file_name}.wav")


import requests, json, time

json_file = "calls/testing/responses.json"
latency_file = "calls/testing/latency.json"

responses = {}
latency = {}

def analysis_stage(folder_path):
    for file in folder_path.iterdir():
        file_name = file.stem
        file_path = folder_path / f"{file_name}.wav"
        print(file_path)

        with open(file_path, "rb") as audio:
            start = time.perf_counter()
            response = requests.post(
                API_URL,
                files={"file": audio}
            )
            api_time = time.perf_counter() - start

        response.raise_for_status()
        responses[int(file_name)] = response.json()
        latency[int(file_name)] = api_time
        print(responses[int(file_name)])
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(latency_file, "r", encoding="utf-8") as f:
            l_data = json.load(f)

        data[file_name] = responses[int(file_name)]
        l_data[file_name] = api_time
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with open(latency_file, "w", encoding="utf-8") as f:
            json.dump(l_data, f, ensure_ascii=False, indent=4)


analysis_stage(preprocessed_folder)


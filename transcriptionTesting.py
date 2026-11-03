import json
dataset_file = "calls/services/dataset.json"
response_file = "calls/testing/responses.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
with open(response_file, "r", encoding="utf-8") as f:
    response_data = json.load(f)

from jiwer import wer, cer

dataset_keys = sorted(dataset.keys(), key=int)
# print(dataset_keys)

full_transcript_gt = ""
full_transcript_response = ""

for key in dataset_keys:
    if key in dataset_keys[22:]:
        full_transcript_gt += dataset[key]["transcript"]
        full_transcript_response += response_data[key]["full_transcription"]
        full_transcript_gt += " "
        full_transcript_response += " "
# print(full_transcript_gt)
# print("----------------------------------------------------")
# print(full_transcript_response)

import re

def normalize_text(text):
    # Remove anything inside (), [], or {} including the brackets
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", "", text)

    # Remove Arabic and English punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text

full_transcript_gt = normalize_text(full_transcript_gt)
full_transcript_response = normalize_text(full_transcript_response)

word_error_rate = wer(full_transcript_gt, full_transcript_response)
character_error_rate = cer(full_transcript_gt, full_transcript_response)


print(f"WER: {word_error_rate:.4f} ({word_error_rate*100:.2f}%)")
print(f"CER: {character_error_rate:.4f} ({character_error_rate*100:.2f}%)")

word_accuracy = (1 - word_error_rate) * 100
character_accuracy = (1 - character_error_rate) * 100

print(f"Word Accuracy: {word_accuracy:.2f}%")
print(f"Character Accuracy: {character_accuracy:.2f}%")
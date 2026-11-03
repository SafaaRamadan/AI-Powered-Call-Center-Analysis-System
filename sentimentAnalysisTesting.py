import json
dataset_file = "calls/services/dataset.json"
response_file = "calls/testing/responses.json"

with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
with open(response_file, "r", encoding="utf-8") as f:
    response_data = json.load(f)


dataset_keys = sorted(dataset.keys(), key=int)

correct = 0
total = 0
skipped = 0

for key in dataset_keys:
    if int(key) not in [7, 9, 10, 11, 21, 27, 26, 33]:
        gt_segments = dataset[key]["segment"]
        pred_segments = response_data[key]["speakers"]

        # Skip if the number of segments differs
        if len(gt_segments) != len(pred_segments):
            # print(key)
            skipped += 1
            # print(
            #     f"Skipped call {key}: "
            #     f"{len(gt_segments)} GT segments vs {len(pred_segments)} predicted."
            # )
            continue

        # Compare sentiment of each corresponding segment
        for gt, pred in zip(gt_segments, pred_segments):

            gt_sentiment = gt["sentiment"].lower()
            pred_sentiment = pred["emotion"].lower()

            if gt_sentiment == pred_sentiment:
                correct += 1
            else:
                print(key)
            total += 1

accuracy = (correct / total) * 100 if total else 0

print(f"\nCompared segments : {total}")
print(f"Correct           : {correct}")
print(f"Skipped calls     : {skipped}")
print(f"Accuracy          : {accuracy:.2f}%")

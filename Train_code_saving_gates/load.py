from datasets import load_dataset
import json

ds = load_dataset("MBZUAI/Bactrian-X", "hi")

ds = ds["train"]
def process_dataset(dataset):
    processed_data = []
    for item in dataset:
        processed_item = item.copy()
        processed_data.append(processed_item)
    return processed_data
print(ds)
train_data = process_dataset(ds)
with open("train_data_hindi.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
import json

with open("data/processed/unified/unified_annotations.json") as f:
    data = json.load(f)

print("Example image entry:", data["images"][0])
print("Example annotation entry:", data["annotations"][0])
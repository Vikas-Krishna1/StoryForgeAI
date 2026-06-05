from datasets import load_dataset
import lzma
import os

os.makedirs("openwebtext", exist_ok=True)

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

num_files = 1000          # adjust as needed
docs_per_file = 100

current_text = []
file_num = 0

for i, sample in enumerate(dataset):
    current_text.append(sample["text"])

    if len(current_text) >= docs_per_file:
        filename = f"openwebtext/{file_num:05d}.xz"

        with lzma.open(filename, "wt", encoding="utf-8") as f:
            f.write("\n".join(current_text))

        print(f"Created {filename}")

        current_text = []
        file_num += 1

        if file_num >= num_files:
            break
    
from datasets import load_dataset
import lzma
import os
from tqdm import tqdm

output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"
files = sorted(os.listdir("openwebtext"))
total_files = len(files)
vocab = set()
spilt_index = int(total_files * 0.9)
files_train = files[:spilt_index]
files_val = files[spilt_index:]
#Proccessing Training FIle
with open(output_file_train, "w", encoding="utf-8") as out_files:
        for filename in (tqdm(files_train, total = len(files_train))):
            with lzma.open(os.path.join("openwebtext", filename), "rt", encoding="utf-8") as in_file:
                text = in_file.read()
                out_files.write(text)
                characters = set(text)
                vocab.update(characters)
      
#Proccessing Validation FIle
with open(output_file_val, "w", encoding="utf-8") as out_files:
        for filename in (tqdm(files_val, total = len(files_val))):
            with lzma.open(os.path.join("openwebtext", filename), "rt", encoding="utf-8") as in_file:
                text = in_file.read()
                out_files.write(text)
                characters = set(text)
                vocab.update(characters)


with open(vocab_file, "w", encoding="utf-8") as vocab_out:
        for char in sorted(vocab):
            vocab_out.write(char + "\n")
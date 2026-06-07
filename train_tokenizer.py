from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[UNK]"]
)

tokenizer.train(
    ["train_split.txt", "val_spilt.txt"],
    trainer
)

tokenizer.save("tokenizer.json")

print("Tokenizer saved.")
print("Vocab size:", tokenizer.get_vocab_size())
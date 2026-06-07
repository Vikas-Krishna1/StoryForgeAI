from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

print(tokenizer.encode("computer").tokens)
print(tokenizer.encode("artificial intelligence").tokens)
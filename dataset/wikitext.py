from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-v1")

with open("wikitext103_train.txt", "w", encoding="utf-8") as f:
    for row in dataset["train"]:
        text = row["text"].strip()
        if text:
            f.write(text + "\n")

print("saved wikitext103_train.txt")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", local_files_only=True)

inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

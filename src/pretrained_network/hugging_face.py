from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("AI models are so smart that they can replace my", max_new_tokens=20))
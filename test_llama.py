from transformers import pipeline

# Load Llama model
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct"
)

result = generator("Explain fake news detection", max_new_tokens=50)

print(result[0]["generated_text"])
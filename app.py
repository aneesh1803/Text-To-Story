!pip install transformers
!pip install torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"  # Using the smaller GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

import torch

def generate_story(prompt, max_length=300):
    # Encode the input prompt to tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a story based on the prompt
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode the generated tokens to text
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Example text prompt to start the story
prompt = "Once upon a time in a faraway kingdom, there was a young princess named Elara who"

# Generate a story
generated_story = generate_story(prompt)
print(generated_story)

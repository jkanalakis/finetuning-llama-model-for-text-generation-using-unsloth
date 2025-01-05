
import torch
from unsloth import FastLanguageModel

# Load the fine-tuned model and tokenizer
model_path = "finetuned_model"  # Path to the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True
)

# Enable optimized inference
FastLanguageModel.for_inference(model)

# Set the model to evaluation mode
model.eval()

# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the appropriate device

# Define a function to generate text based on a given prompt
def generate_text(input_prompt, max_length=200, temperature=0.7, top_p=0.9):
    """Generate text using the fine-tuned model.

    Args:
        input_prompt (str): The input prompt for text generation.
        max_length (int): Maximum length of the generated text.
        temperature (float): Sampling temperature for generation.
        top_p (float): Top-p nucleus sampling parameter.

    Returns:
        str: The generated text.
    """
    # Tokenize the input prompt
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    # Ensure model and inputs are on the same device
    assert model.device == inputs["input_ids"].device, "Model and inputs are on different devices."

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
        )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print("Generating text...")
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)

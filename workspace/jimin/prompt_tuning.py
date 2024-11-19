
from transformers import LLaVAModel, LLaVATokenizer
import torch

# Replace with your local model path
local_model_path = "/home/work/yaiq/datasets/llava-v1.6-vicuna-13b-hf"

# Load the LLaVA model and tokenizer
model = LLaVAModel.from_pretrained(local_model_path)
tokenizer = LLaVATokenizer.from_pretrained(local_model_path)

# Check model device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define learnable prompt embeddings
prompt_length = 10  # Adjust based on task complexity
prompt_embeddings = torch.nn.Parameter(
    torch.randn(prompt_length, model.config.hidden_size, requires_grad=True)
)

# Optimizer for prompt tuning
optimizer = torch.optim.Adam([prompt_embeddings], lr=5e-5)


def preprocess_data(images, texts, labels):
    # Tokenize texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Add learnable prompts to input embeddings
    input_embeds = inputs.input_ids.to(device)  # Token IDs
    image_tensors = images.to(device)  # Move images to GPU/CPU
    labels = labels.to(device)  # Move labels to GPU/CPU

    return {"images": image_tensors, "text": input_embeds, "labels": labels}


for param in model.parameters():
    param.requires_grad = False  # Freeze LLaVA model parameters

num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:  # Replace with your data loader
        images, texts, labels = batch

        # Preprocess inputs
        inputs = preprocess_data(images, texts, labels)
        images, text_inputs, labels = (
            inputs["images"],
            inputs["text"],
            inputs["labels"],
        )

        # Forward pass
        outputs = model(image=images, text=text_inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)  # Define appropriate loss function

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")




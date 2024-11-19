from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from PIL import Image
import torch

# Load the model and tokenizer
model_name = "llava-v1.6-vicuna-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir = "/home/work/yaiq/datasets/llava-v1.6-vicuna-13b-hf")


# Load CSV file with image paths and associated prompts
csv_path = "path_to_your_csv.csv"
data = pd.read_csv(csv_path)

# Define five text prompts you want to use
text_prompts = [
    "Prompt type 1: {}",
    "Prompt type 2: {}",
    "Prompt type 3: {}",
    "Prompt type 4: {}",
    "Prompt type 5: {}"
]

results = []

for index, row in data.iterrows():
    image_path = row["image_column_name"]  # Replace with actual column name for images
    image = Image.open(image_path).convert("RGB")
    
    # Generate output for each prompt
    for prompt_template in text_prompts:
        text_prompt = prompt_template.format(row["text_column_name"])  # Column for custom text if needed
        inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
        
        # Combine image and text input as model requires
        # Assuming the model supports image encoding (consult model's doc if specific formatting is needed)
        image_inputs = tokenizer(images=image, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, **image_inputs)
        
        # Decode and store results
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "image_path": image_path,
            "text_prompt": text_prompt,
            "output": result_text
        })


results_df = pd.DataFrame(results)
results_df.to_csv("multimodal_inference_results.csv", index=False)


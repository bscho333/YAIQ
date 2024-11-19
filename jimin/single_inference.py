from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import pandas as pd
from PIL import Image
import torch

# Ensure bitsandbytes is installed for 8-bit loading
# pip install bitsandbytes

# Load the model and processor with 8-bit quantization and memory-efficient settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,       # Use 8-bit precision to reduce memory
    llm_int8_enable_fp32_cpu_offload=True  # Offload FP32 layers to CPU if needed
)

model = AutoModelForImageTextToText.from_pretrained(
    "/home/work/yaiq/datasets/llava-v1.6-vicuna-13b-hf",
    quantization_config=quantization_config,
    device_map="auto"  # Automatically spread model across available resources
)

processor = AutoProcessor.from_pretrained("/home/work/yaiq/datasets/llava-v1.6-vicuna-13b-hf")

# Load CSV file with image paths and associated prompts
csv_path = "/home/work/yaiq/datasets/dataset50.csv"
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
default_dataset_path = "/home/work/yaiq/datasets"

for index, row in data.iterrows():
    data_path = f"{default_dataset_path}/{row['path']}/"  # Adjust to actual column name for images
    question_image_path = f"{data_path}question/image/question.jpeg"
    question_image = Image.open(question_image_path).convert("RGB")
    
    # Generate output for each prompt
    for prompt_template in text_prompts:
        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "What is shown in this image?"},
        #             {"type": "image"},
        #         ],
        #     }
        # ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=question_image, text=prompt, return_tensors="pt").to(device)
        
        # Autoregressively complete the prompt
        with torch.no_grad():  # Save memory by not calculating gradients
            outputs = model.generate(**inputs)
        
        # Decode and store results
        result_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "image_path": question_image_path,
            "text_prompt": prompt,
            "output": result_text
        })
        print(results[-1])

# Uncomment to save results to a CSV file
# results_df = pd.DataFrame(results)
# results_df.to_csv("multimodal_inference_results.csv", index=False)

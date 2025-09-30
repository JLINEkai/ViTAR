import io
import json
import os
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
from PIL import Image
from tqdm import tqdm

from verl.workers.agent.envs.mm_process_engine.visual_toolcrop import \
    VisualToolCrop

with open('medical_multimodel_evaluation_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

client = openai.OpenAI(
    base_url="http://localhost:8003/v1",  
    api_key="EMPTY" 
)
model_name = "/mnt/shared-storage-user/checkpoint/ViTAR/actor/huggingface"  


def convert_to_full_path(img_field):

    if img_field.startswith("images/"):
        return os.path.join("./MedicalDataset/Medical_Multimodal_Evaluation_Data", img_field)  
    else:
        return os.path.join("XXX", img_field)
system_prompt = """You are a medical image analysis assistant capable of analyzing medical images and answering questions about them. Your goal is to answer questions about medical images including modality, body part, and other medical details. You can rely on your own capabilities or use mark tools to assist in solving.\nYour output should be in a strict JSON format as follows:\n{\"thought\": \"the reasoning process\", \"actions\": [{\"name\": \"action\", \"arguments\": {\"argument1\": \"value1\"}}]}"""

def send_request(img_path, text):
    tool = VisualToolCrop("visual_toolcrop", "Tool for image cropping", {})


    if not (img_path.startswith("http://") or img_path.startswith("https://") or img_path.startswith("data:") or img_path.startswith("file://")):
        img_path = "file://" + img_path

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt}
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_path}},
            {"type": "text", "text": text}
        ]}
    ]
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
        )
        response = chat_response.choices[0].message.content.strip()

        if img_path.startswith("file://"):
            pil_img = Image.open(img_path[7:])
        else:
            pil_img = Image.open(requests.get(img_path, stream=True).raw)

        tool.reset(raw_prompt=None, multi_modal_data={'image': [pil_img]}, origin_multi_modal_data={'image': [pil_img]})
        obs, reward, done, info = tool.execute(response)
       
        cropped_img = obs['multi_modal_data']['image'][0]
        save_dir = "./data/cropped_images"
        os.makedirs(save_dir, exist_ok=True)
        unique_name = f"crop_{uuid.uuid4().hex}.jpg"
        if cropped_img.mode == 'RGBA':
            background = Image.new('RGB', cropped_img.size, (255, 255, 255))
            background.paste(cropped_img, mask=cropped_img.split()[-1]) 
            cropped_img = background
        save_path = os.path.join(save_dir, unique_name)
        cropped_img.save(save_path)
        file_url = f"file://{save_path}"
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": system_prompt}
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_path}},
                {"type": "text", "text": text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": response}
            ]}
        ]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "I have marked the region of interest on the image. Please answer the question based on the marked region."},
                {"type": "image_url", "image_url": {"url": file_url}}
            ]
        })
        
        chat_response2 = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
        )
        response2 = chat_response2.choices[0].message.content.strip()

        return {"first_response": response, "tool_obs": str(obs), "second_response": response2}

    except Exception as e:
        return f"Error: {str(e)}"


qa_pairs = []
for item in data:
    img_list = item.get("image", [])
    img_path = img_list[0] if img_list else ""
    img_path = convert_to_full_path(img_path) if img_path else ""
    text = item.get("question", "")
    ground_truth = item.get("answer", "")
    options = item.get("options", [])
    dataset = item.get("dataset", "")
    subset = item.get("subset", "")
    
    if options:
        option_str = "\nOptions:\n" + "\n".join([f"{chr(65+i)} {opt}" for i, opt in enumerate(options)])
        text = text.strip() + option_str
    
    if img_path and text:
        ground_truth_letter = ""
        if ground_truth and options:
            try:
                answer_index = options.index(ground_truth)
                ground_truth_letter = chr(65 + answer_index) 
            except ValueError:
                ground_truth_letter = ground_truth
        
        qa_pairs.append({
            "image": img_path,
            "question": text,
            "ground_truth": ground_truth_letter,
            "ground_truth_original": ground_truth, 
            "dataset": dataset,
            "subset": subset,
            "options": options
        })

print(f"总共找到 {len(qa_pairs)} 个问答对")

results = []
with ThreadPoolExecutor(max_workers=256) as executor:
    future_to_info = {}
    for qa in qa_pairs:
        future = executor.submit(send_request, qa["image"], qa["question"])
        future_to_info[future] = qa
    
    with tqdm(total=len(qa_pairs), desc="处理问题") as pbar:
        for future in as_completed(future_to_info):
            qa = future_to_info[future]
            result = future.result()
            qa["answer"] = result
            results.append(qa)
            pbar.update(1)

# 保存结果
with open('output_v923_token.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"共处理 {len(results)} 个问题")

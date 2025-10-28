import numpy as np
from typing import List, Dict
import torch
from PIL import Image
import io
import time
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from helper import device, load_json_variable
import streamlit as st

class Summary:
    def __init__(self, model_components):
        self.model_components = model_components

    def generate_summary(self, detections, object_counts, image: Image.Image, ):

        if not self.model_components or not self.model_components['processor']:
            return "Model not loaded properly"
        
        # print("Detections:", detections)
        
        prompt = f"You are a security surveillance AI. Analyze this image and the detection data: {detections}. Total objects detected: {object_counts}. Provide a brief security assessment in the format: '[Number] [objects] detected [location/activity] — [security assessment/concern]'. Focus on potential security issues, unusual activities, or normal situations. Examples: '1 person detected in bedroom during daytime — normal activity', '3 people detected near entrance after hours — possible security concern', '1 person detected in restricted area — unauthorized access alert'."

        try:
            processor = self.model_components['processor']
            model = self.model_components['model']
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_for_model = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt_for_model, images=[image], return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=load_json_variable("max_token"),
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=True
                )
            
            generated_text = processor.decode(output[0], skip_special_tokens=True)
            
            if "assistant\n" in generated_text:
                response = generated_text.split("assistant\n")[-1].strip()
            elif "[/INST]" in generated_text:
                response = generated_text.split("[/INST]")[-1].strip()
            else:
                response = generated_text.strip()

            # print(response)
            
            return response
            
        except Exception as e:
            return f"Error processing: {str(e)}"

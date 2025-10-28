import json
import cv2 as cv
import torch
from vision import Vision
from summary import Summary
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from helper import device
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from helper import load_json_variable

def load_llava_model():
    disable_torch_init()
    model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")
        
        return {
            'tokenizer': processor.tokenizer,
            'model': model,
            'image_processor': processor.image_processor,
            'context_len': 4096,
            'processor': processor
        }
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        return None

if __name__ == "__main__":
    
    llava_model_components = load_llava_model()
    
    if llava_model_components:
        summary_model = Summary(llava_model_components)
        print("Summary model loaded.")

        vision_system = Vision(model_name=load_json_variable("model"), 
                               confidence_threshold=load_json_variable("confidence_threshold"), 
                               stream=load_json_variable("stream"), 
                               summary_model=summary_model)
        video_source = 0
        vision_system.process_viewpoint(video_source)
        print("Detections: In main")
        
        
    else:
        print("Could not load LLaVA model, exiting.")    
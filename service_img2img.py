import gradio as gr

import numpy as np 
import cv2
from PIL import Image, ImageDraw
import torch 
from torch import autocast

from segment_anything import build_sam, SamAutomaticMaskGenerator 
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from utils import segment_image, convert_box_xywh_to_xyxy 

from diffusers import StableDiffusionLongPromptWeightingPipeline, EulerDiscreteScheduler
from torch import autocast

from diffusers import DPMSolverMultistepScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth").to(device))
print('load segement anything model.')

model = CLIPModel.from_pretrained("clip")
processor = CLIPProcessor.from_pretrained("clip")
model.to(device)
print('load clip model.')


model_id = "waifu-research-department/long-prompt-weighting-pipeline"
pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None

pipe = pipe.to(device) 
print('load sd model.')

negative_prompts = [
    "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low-res, normal quality, ((monochrome)), ((grayscale)), skin spots, acne, skin blemishes, age spots, glans"
]

def image_resize(img):
    width, height = img.size
    print(width, height)
    left = width // 2 - height // 2
    right = width // 2 + height // 2

    top = 0
    bottom = height 

    img = img.crop((left, top, right, bottom)) 
    new_size = (512, 512)
    img = img.resize(new_size) 
    return img

@torch.no_grad()
def retriev(elements, search_text):
    preprocessed_images = processor(images=elements, return_tensors="pt")
    tokenized_text = processor(text = [search_text], padding=True, return_tensors="pt")

    print(preprocessed_images, tokenized_text)

    preprocessed_images['pixel_values'] = preprocessed_images['pixel_values'].to(device) 
    tokenized_text['input_ids'] = tokenized_text['input_ids'].to(device) 
    tokenized_text['attention_mask'] = tokenized_text['attention_mask'].to(device) 

    image_features = model.get_image_features(**preprocessed_images)
    text_features = model.get_text_features(**tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0) 


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]



def segment(
    clip_threshold: float,
    image_path: str,
    segment_query: str,
    text_prompt: str,
):
    image = Image.open(image_path) 
    image = image_resize(image)
    image.save(image_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    image = Image.open(image_path)
    #image = image_resize(image)
    cropped_boxes = []

    for mask in tqdm(masks):
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"]))) 

    scores = retriev(cropped_boxes, segment_query)
    indices = get_indices_of_values_above_threshold(scores, clip_threshold)

    segmentation_masks = []

    for seg_idx in indices:
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        segmentation_masks.append(segmentation_mask_image)

    original_image = Image.open(image_path)
    #original_image = image_resize(original_image)
    # overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 255)) #0))
    # overlay_color = (255, 255, 255, 0) #0, 0, 0, 200)
    overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
    overlay_color = (255, 255, 255, 0)

    draw = ImageDraw.Draw(overlay_image)
    for segmentation_mask_image in segmentation_masks:
        draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

    # return Image.alpha_composite(original_image.convert('RGBA'), overlay_image) 
    mask_image = overlay_image.convert("RGB") 
    
    #with autocast("cuda"):
    #gen_image = sd_pipe(prompt=text_prompt, image=original_image, mask_image=mask_image).images[0] 
    with autocast("cuda"):
        gen_image = pipe(text_prompt, image=original_image, mask_image =mask_image, negative_prompt = '', guidance_scale=10, num_inference_steps=30, height=512, width=512).images[0]
    #target = Image.new("RGB", (512 * 2, 512))
    #target.paste(mask_image, (0, 0)) 
    #target.paste(gen_image, (0, 512)) 
    return mask_image, gen_image 



demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Slider(0, 1, value=0.05, label="clip_threshold"),
        gr.Image(type="filepath"),
        "text",
        "text",
    ],
    outputs=["image", "image"],
    allow_flagging="never",
    title="Segment Anything Model with Stable Diffusion Model",
)

if __name__ == "__main__":
    demo.launch(enable_queue=True, server_name='0.0.0.0',server_port=8413)

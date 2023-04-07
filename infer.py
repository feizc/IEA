import numpy as np 
import cv2
from PIL import Image, ImageDraw
import torch 

from segment_anything import build_sam, SamAutomaticMaskGenerator 
from transformers import CLIPProcessor, CLIPModel

from utils import segment_image, convert_box_xywh_to_xyxy 

mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))

image_path = "cat.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

image = Image.open(image_path)
cropped_boxes = []

for mask in masks:
    cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"]))) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("clip")
processor = CLIPProcessor.from_pretrained("clip")

print('clip')
@torch.no_grad()
def retriev(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = processor(images=elements, return_tensors="pt")
    tokenized_text = processor(text = [search_text], padding=True, return_tensors="pt")

    image_features = model.get_image_features(**preprocessed_images)
    text_features = model.get_text_features(**tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0) 


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

scores = retriev(cropped_boxes, "cat")
indices = get_indices_of_values_above_threshold(scores, 0.05)

segmentation_masks = []

for seg_idx in indices:
    segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
    segmentation_masks.append(segmentation_mask_image)

original_image = Image.open(image_path)
overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
overlay_color = (255, 0, 0, 200)

draw = ImageDraw.Draw(overlay_image)
for segmentation_mask_image in segmentation_masks:
    draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)
result_image.save('cut.png')
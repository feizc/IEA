# IEA: Image Editing Anything


Using stable diffusion and segmentation anything models for image editing. 

Generally, given a textual prompt or cliked region, SAM generated the masked region for source image. Then, we use CLIP model to select the region, which is then used to generate the target edited image with stable diffusion.

Use ```python service.py``` to initialize the service. 




# Reference 

[1] https://github.com/huggingface/diffusers 

[2] https://github.com/facebookresearch/segment-anything

[3] https://github.com/maxi-w/CLIP-SAM

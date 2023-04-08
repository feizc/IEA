# IEA: Image Editing Anything


Using stable diffusion and segmentation anything models for image editing. 

Generally, given a textual prompt or cliked region, SAM generated the masked region for source image. Then, we use CLIP model to select the region, which is then used to generate the target edited image with stable diffusion.

Use ```python service.py``` to initialize the service. 

## Generated Cases

<img width="810" alt="case" src="https://user-images.githubusercontent.com/37614046/230707537-206c0714-de32-41cd-a277-203fd57cd300.png">

<img width="808" alt="116f87dc-0232-4773-b5e7-98691426f915" src="https://user-images.githubusercontent.com/37614046/230707944-2bac30ca-4fce-4fcf-84d1-71f52fe8d2d4.png">



# Reference 

[1] https://github.com/huggingface/diffusers 

[2] https://github.com/facebookresearch/segment-anything

[3] https://github.com/maxi-w/CLIP-SAM

[4] https://github.com/IDEA-Research/Grounded-Segment-Anything/

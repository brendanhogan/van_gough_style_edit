# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from pathlib import Path as pth
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline
from controlnet_aux import OpenposeDetector

import cv2
import requests


from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def setup_pipeline():
    """
    Setup full diffusion pipeline, with img2img, canny, depth, and pose conrol nets 
    """

    # Model pths 
    pip_2_pix_path = "lllyasviel/control_v11e_sd15_ip2p"
    depth_path = "lllyasviel/control_v11f1p_sd15_depth"
    canny_path = "lllyasviel/control_v11p_sd15_canny"
    pose_path = "lllyasviel/sd-controlnet-openpose"


    # Control pipes 
    pix_2_pix_control = ControlNetModel.from_pretrained(pip_2_pix_path, torch_dtype=torch.float16)
    depth_control = ControlNetModel.from_pretrained(depth_path, torch_dtype=torch.float16)
    canny_control = ControlNetModel.from_pretrained(canny_path, torch_dtype=torch.float16)
    pose_control = ControlNetModel.from_pretrained(pose_path, torch_dtype=torch.float16)
    control_nets = [pix_2_pix_control, depth_control, canny_control,pose_control]

    # Final pipeline 
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=control_nets, torch_dtype=torch.float16
    ).to("cuda")

    # Better memory usage 
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    return pipe

def setup_upscaler():
    """
    Setup control net to used for high res upscaling 
    """
    controlnet = ControlNetModel.from_pretrained('models/tile_upscale_controlnet', torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",custom_pipeline="stable_diffusion_controlnet_img2img",controlnet=controlnet,torch_dtype=torch.float16).to('cuda')
    # pipe.enable_xformers_memory_efficient_attention()

    return pipe 

def load_image(image):
    """
    Load image from path or url to PIL image 
    """ 
    if isinstance(image, str):
        if image.startswith("http"):
            image = Image.open(requests.get(image, stream=True).raw)
        else:
            image = Image.open(image)
    elif isinstance(image, pth):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type {type(image)}")

    return image


def get_control_images(image, depth_estimator, pose_detector):
    """
    Setup the img2imgs, depth, canny, and pose control images 
    """

    # Img2img control image (just original image)
    img2img_control = image

    # Depth control image
    depth_control_image = depth_estimator(image)['depth']
    depth_control_image = np.array(depth_control_image)
    depth_control_image = depth_control_image[:, :, None]
    depth_control_image = np.concatenate([depth_control_image, depth_control_image, depth_control_image], axis=2)
    control_depth_control_image = Image.fromarray(depth_control_image)

    # Canny control image
    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(np.array(image), low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    # Pose control image
    pose_image = pose_detector(image)

    # All images as list 
    input_images = [img2img_control, control_depth_control_image, canny_image, pose_image]

    return input_images

def resize_for_condition_image(input_image, resolution):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def upscale_image(upscale_pipe,image):
    # Set image to desired size  
    condition_image = resize_for_condition_image(image, 1024)

    # Run stable diffusion
    image = upscale_pipe(prompt="best quality, ultra quality, 4k, 6k, sharp, crisp, smooth", 
                negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
                image=condition_image, 
                controlnet_conditioning_image=condition_image, 
                width=condition_image.size[0],
                height=condition_image.size[1],
                strength=1.0,
                generator=torch.manual_seed(0),
                num_inference_steps=32,
                ).images[0]
    
    return image  

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # Setup pipeline and upscaler 
        self.pipe = setup_pipeline() 
        self.upscaler = setup_upscaler()

        # Setup depth estimatior and pose detector
        self.depth_estimator = pipeline('depth-estimation')
        self.pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    def predict(
        self,
        image_pth: Path = Input(description="Link or path to image to process"),

    ) -> Path:
        """Run a single prediction on the model"""
        # 1. Download image from url, and resize 
        image = Image.open(image_pth)
        image = image.resize((512, 512))

        # 2. Setup all control images 
        control_images = get_control_images(image, self.depth_estimator, self.pose_detector)

        # 3. Setup parameters for run (weights, prompts - these need to be tuned)
        control_net_scales = [.1,.3,.55,.4]
        prompt = "a van gough style painting of a person, swirling colors, texture, van gough style, ultra quality, sharp focus, tack sharp, dof, film grain, crystal clear,van gough style 8K UHD, highly detailed glossy eyes, high detailed skin, artistic"
        negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"

        # 4. Run control net pipe 
        output_image = self.pipe(prompt,control_images, num_inference_steps=30, generator=torch.Generator(device="cpu").manual_seed(1), negative_prompt=negative_prompt,controlnet_conditioning_scale=control_net_scales).images[0]

        # 5. Upscale image
        output_image_high_res = upscale_image(self.upscaler,output_image)

        # 6. Return image 
        output_path = f"/tmp/high_res_output.png"
        output_image_high_res.save(output_path)

        return Path(output_path)


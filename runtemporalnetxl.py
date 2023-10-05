import os
import cv2
import torch
import argparse
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
from PIL import Image

def split_video_into_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    print("splitting video")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(frames_dir, f"frame{count:04d}.png")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1

def frame_number(frame_filename):
    # Extract the frame number from the filename and convert it to an integer
    return int(frame_filename[5:-4])

def count_frame_images(frames_dir):
    # Count the number of frame images in the directory
    frame_files = [f for f in os.listdir(frames_dir) if f.startswith('frame') and f.endswith('.png')]
    return len(frame_files)

# Argument parser
parser = argparse.ArgumentParser(description='Generate images based on video frames.')
parser.add_argument('--prompt', default='a woman', help='the stable diffusion prompt')
parser.add_argument('--video_path', default='./None.mp4', help='Path to the input video file.')
parser.add_argument('--frames_dir', default='./frames', help='Directory to save the extracted video frames.')
parser.add_argument('--output_frames_dir', default='./output_frames', help='Directory to save the generated images.')
parser.add_argument('--init_image_path', default=None, help='Path to the initial conditioning image.')

args = parser.parse_args()

video_path = args.video_path
frames_dir = args.frames_dir
output_frames_dir = args.output_frames_dir
init_image_path = args.init_image_path
prompt = args.prompt

# If frame images do not already exist, split video into frames
if count_frame_images(frames_dir) == 0:
    split_video_into_frames(video_path, frames_dir)

# Create output frames directory if it doesn't exist
if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# Load the initial conditioning image, if provided
if init_image_path:
    print(f"using image {init_image_path}")
    last_generated_image = load_image(init_image_path)
else:
    initial_frame_path = os.path.join(frames_dir, "frame0000.png")
    last_generated_image = load_image(initial_frame_path)

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet1_path = "CiaraRowles/controlnet-temporalnet-sdxl-1.0"
controlnet2_path = "diffusers/controlnet-canny-sdxl-1.0"

controlnet = [
    ControlNetModel.from_pretrained(controlnet1_path, torch_dtype=torch.float16,use_safetensors=True),
    ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)
]
#controlnet = ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(7)

# Loop over the saved frames in numerical order
frame_files = sorted(os.listdir(frames_dir), key=frame_number)

for i, frame_file in enumerate(frame_files):
    # Use the original video frame to create Canny edge-detected image as the conditioning image for the first ControlNetModel
    control_image_path = os.path.join(frames_dir, frame_file)
    control_image = load_image(control_image_path)
    
    canny_image = np.array(control_image)
    canny_image = cv2.Canny(canny_image, 25, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    # Generate image
    image = pipe(
       prompt, num_inference_steps=20, generator=generator, image=[last_generated_image, canny_image], controlnet_conditioning_scale=[0.6, 0.7]
       #prompt, num_inference_steps=20, generator=generator, image=canny_image, controlnet_conditioning_scale=0.5
    ).images[0]
    
    # Save the generated image to output folder
    output_path = os.path.join(output_frames_dir, f"output{str(i).zfill(4)}.png")
    image.save(output_path)

    # Save the Canny image for reference
    canny_image_path = os.path.join(output_frames_dir, f"outputcanny{str(i).zfill(4)}.png")
    canny_image.save(canny_image_path)

    # Update the last_generated_image with the newly generated image for the next iteration
    last_generated_image = image

    print(f"Saved generated image for frame {i} to {output_path}")


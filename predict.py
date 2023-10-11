# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import Optional, List

import os
import subprocess

os.environ["HUGGINGFACE_HUB_CACHE"] = "/dev/null"

import cv2
import torch
import shutil
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
from PIL import Image

from weights import download_all_weights

# files to download from the weights mirrors
WEIGHT_FILES = [
    {
        "dest": "stabilityai/stable-diffusion-xl-base-1.0",
        "src": "sdxl-base-1.0/stable-diffusion-xl-base-1.0",
        "files": [
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "model_index.json",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "vae_1_0/diffusion_pytorch_model.safetensors",
            "tokenizer_2/merges.txt",
            "tokenizer_2/vocab.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json"
        ]
    },
    {
        "dest": "CiaraRowles/controlnet-temporalnet-sdxl-1.0",
        "src": "CiaraRowles--controlnet-temporalnet-sdxl-1.0/1ec451603dbca5ad0a1e2dd80db71dbaa308c073",
        "files": [
            "config.json",
            "diffusion_pytorch_model.safetensors"
        ]
    },
    {
        "dest": "diffusers/controlnet-canny-sdxl-1.0",
        "src": "diffusers--controlnet-canny-sdxl-1.0/eb115a19a10d14909256db740ed109532ab1483c",
        "files": [
            "config.json",
            "diffusion_pytorch_model.safetensors"
        ]
    }
]

def split_video_into_frames(video_path, frames_dir, max_count=None):

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
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
        if max_count == count:
            return

def frame_number(frame_filename):
    # Extract the frame number from the filename and convert it to an integer
    return int(frame_filename[5:-4])


def get_fps(reference_video):
    cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {reference_video}"
    fps = subprocess.check_output(cmd, shell=True).decode().strip()
    num, den = map(int, fps.split('/'))
    return num / den

def images_to_video(image_prefix, output_video, reference_video):
    fps = get_fps(reference_video)
    cmd = f"ffmpeg -framerate {fps} -i {image_prefix}%04d.png {output_video}"
    subprocess.run(cmd, shell=True)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        download_all_weights(WEIGHT_FILES, to="weights")
        # self.model = torch.load("./weights.pth")

        base_model_path = "weights/stabilityai/stable-diffusion-xl-base-1.0"
        controlnet1_path = "weights/CiaraRowles/controlnet-temporalnet-sdxl-1.0"
        controlnet2_path = "weights/diffusers/controlnet-canny-sdxl-1.0"

        controlnet = [
            ControlNetModel.from_pretrained(controlnet1_path, torch_dtype=torch.float16,use_safetensors=True),
            ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)
        ]
        #controlnet = ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16
        )

        #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        #pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        prompt: str = Input(description="The stable diffusion prompt. Does what it says on the tin."),
        video: Path = Input(description="The input video file."),
        init_image_path: Path = Input(description="Path to the initial conditioning image. It is recommended you get the first frame, modify it to a good starting look with stable diffusion, and use that as the first generated frame, if unspecified it will use the first video frame (not recommended)", default=None),
        seed: int = Input(description="Seed. Use this to get the same result", default=None),
        max_frames: int = Input(description="Only use the first N frames of the output video. 0 to use all frames.", default=0, ge=0),
        result_video: bool = Input(description="Return the output as a video. Otherwise, all frames are returned separately.", default=True)
    ) -> List[Path]:
        """Run a single prediction on the model"""
        frames_dir = "./frames"
        output_frames_dir = "./output_frames"
        split_video_into_frames(str(video), frames_dir, max_frames if max_frames != 0 else None)
        
        # Clear and create output frames directory
        if os.path.exists(output_frames_dir):
            shutil.rmtree(output_frames_dir)
        os.makedirs(output_frames_dir)

        # Load the initial conditioning image, if provided
        if init_image_path is not None:
            print(f"using image {str(init_image_path)}")
            last_generated_image = load_image(str(init_image_path))
        else:
            initial_frame_path = os.path.join(frames_dir, "frame0000.png")
            last_generated_image = load_image(initial_frame_path)

    
        generator = torch.Generator() if seed is None else torch.manual_seed(seed)

        # Loop over the saved frames in numerical order
        frame_files = sorted(os.listdir(frames_dir), key=frame_number)
        output_files = []
        output_files_canny = []
        
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
            image = self.pipe(
                prompt, num_inference_steps=20, generator=generator, image=[last_generated_image, canny_image], controlnet_conditioning_scale=[0.6, 0.7]
                #prompt, num_inference_steps=20, generator=generator, image=canny_image, controlnet_conditioning_scale=0.5
            ).images[0]
            
            # Save the generated image to output folder
            output_path = os.path.join(output_frames_dir, f"output{str(i).zfill(4)}.png")
            image.save(output_path)
            output_files.append(Path(output_path))
            
            # Save the Canny image for reference
            canny_image_path = os.path.join(output_frames_dir, f"outputcanny{str(i).zfill(4)}.png")
            canny_image.save(canny_image_path)
            output_files_canny.append(Path(canny_image_path))
            
            # Update the last_generated_image with the newly generated image for the next iteration
            last_generated_image = image
            
            print(f"Saved generated image for frame {i} to {output_path}")

        if result_video:
            images_to_video(os.path.join(output_frames_dir, "output"), "output.mp4", str(video))
            images_to_video(os.path.join(output_frames_dir, "outputcanny"), "output_canny.mp4", str(video))
            return [Path("output.mp4"), Path("output_canny.mp4")]
        else:
            return output_files + output_files_canny


---
license: openrail++
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---

[![Replicate](https://replicate.com/yorickvp/temporalnet-xl/badge)](https://replicate.com/yorickvp/temporalnet-xl) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-yellow)](https://huggingface.co/CiaraRowles/controlnet-temporalnet-sdxl-1.0)
    
# TemporalNetXL

This is TemporalNet1XL, it is a re-train of the controlnet TemporalNet1 with Stable Diffusion XL.

This does not use the control mechanism of TemporalNet2 as it would require some additional work to adapt the diffusers pipeline to work with a 6-channel input.

In order to run, simply use the script "runtemporalnetxl.py" after installing the normal diffusers requirements and specify the following command line arguments:

--prompt  does what it says on the tin

--video_path the path to your input video, this will split the frames out if the frames are not already there, if you want a different resolution or frame rate, you'll want to preprocess them and put them into the ./frames folder

--frames_dir (optional) if you want a different path for the frames input

--output_frames_dir (optional) the output directory

--init_image_path (optional) it is recommended you get the first frame, modify it to a good starting look with stable diffusion, and use that as the first generated frame, if unspecified it will use the first video frame (not recommended)

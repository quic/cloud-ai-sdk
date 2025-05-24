# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import asyncio
import os
import torch

from diffusers import AutoPipelineForText2Image

class QAICStableDiffusion:
    def __init__(self, model_id = 'stabilityai/sdxl-turbo', device_id=0):
        text_encoder = './qpc/text_encoder_256b_512i_16c_1b/programqpc.bin'
        unet = './qpc/unet_256b_512i_16c_1b_2m_1o/programqpc.bin'
        text_encoder_2 = './qpc/text_encoder_2_256b_512i_16c_1b/programqpc.bin'
        sdxl_vae_decoder = './qpc/vae_decoder_256b_512i_vae_16c_1b_2m_1o/programqpc.bin'

        # check the QPCs
        unet_qpc = unet if unet.endswith('programqpc.bin') else os.path.join(unet,'programqpc.bin')
        assert os.path.isfile(unet_qpc), f"Could not find binary {unet_qpc = }!"
        vae_decoder_sdxl_qpc = sdxl_vae_decoder if sdxl_vae_decoder.endswith('programqpc.bin') else os.path.join(sdxl_vae_decoder,'programqpc.bin')
        assert os.path.isfile(vae_decoder_sdxl_qpc), f"Could not find binary {vae_decoder_sdxl_qpc = }!"
        text_encoder_qpc = text_encoder if text_encoder.endswith('programqpc.bin') else os.path.join(text_encoder,'programqpc.bin')
        assert os.path.isfile(text_encoder_qpc), f"Could not find binary {text_encoder_qpc = }!"
        text_encoder_2_qpc = text_encoder_2 if text_encoder_2.endswith('programqpc.bin') else os.path.join(text_encoder_2,'programqpc.bin')
        assert os.path.isfile(text_encoder_2_qpc), f"Could not find binary {text_encoder_2_qpc = }!"

        self.num_steps = 1
        self.vae_type = "vae"

        # load the latents
        self.latents = None

        # load the model pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16",
                                                device_id=device_id,
                                                unet_qpc=unet_qpc,
                                                vae_decoder_qpc=vae_decoder_sdxl_qpc,
                                                text_encoder_qpc=text_encoder_qpc,
                                                text_encoder_2_qpc=text_encoder_2_qpc)

    async def generate(self, prompt, n, image_size):
        height, width = image_size[0], image_size[1]
        images = self.pipe(prompt=prompt,
                    num_inference_steps=self.num_steps,
                    height=height,
                    width=width,
                    latents=self.latents,
                    vae_type=self.vae_type,
                    guidance_scale=0.0).images

        yield images[0]

async def main():
    model = QAICStableDiffusion()
    prompt = 'A cinematic shot of a baby racoon wearing an intricate italian priest robe.'
    idx = 0
    async for image in model.generate(prompt, 1, (512, 512)):
       image.save('generated_image_{}.png'.format(idx))
       idx += 1

if __name__ == "__main__":
    asyncio.run(main())


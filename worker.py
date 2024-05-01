from diffusers import AutoPipelineForText2Image
import torch
import json

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    requires_safety_checker=False).to("cuda:1")

import gradio as gr

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def is_parsable_json(command):
    try:
        json.loads(command)
        return True
    except json.JSONDecodeError:
        return False

def generate(command):
    if is_parsable_json(command):
        values = json.loads(command)
        width = closestNumber(values['width'], 8)
        height = closestNumber(values['height'], 8)
        image = pipe(values['prompt'], negative_prompt=values['negative_prompt'], num_inference_steps=1, guidance_scale=0.0, width=width, height=height).images[0]
        image.save('/content/image.jpg')
        return image
    else:
        width = closestNumber(512, 8)
        height = closestNumber(512, 8)
        image = pipe(command, num_inference_steps=1, guidance_scale=0.0, width=width, height=height).images[0]
        image.save('/content/image.jpg')
        return image

with gr.Blocks(title=f"sdxl-turbo", css=".gradio-container {max-width: 544px !important}", analytics_enabled=False) as demo:
    with gr.Row():
      with gr.Column():
          textbox = gr.Textbox(show_label=False, value="a close-up picture of a fluffy cat")
          button = gr.Button()
    with gr.Row(variant="default"):
        output_image = gr.Image(
            show_label=False,
            type="pil",
            interactive=False,
            height=512,
            width=512,
            elem_id="output_image",
        )

    button.click(fn=generate, inputs=[textbox], outputs=[output_image], show_progress=False)

import os
PORT = int(os.getenv('server_port'))
demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=PORT)
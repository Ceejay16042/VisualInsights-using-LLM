#libraries
from diffusers import StableDiffusionPipeline
import torch

device = 'cuda' if torch.cuda.is_available else "cpu"


#Image Generation LLM
def CGI(prompt):
  model_id = "runwayml/stable-diffusion-v1-5"
  pipe = StableDiffusionPipeline.from_pretrained(model_id,
  torch_dtype=torch.float16).to(device)
  # pipe = pipe.to(device)

  prompt_instruction = prompt
  image = pipe(prompt).images[0]
  return image

#libraries
from gtts import gTTS
import re
import os
from PIL import Image
import time
import torch
from transformers import  pipeline, BitsAndBytesConfig
import warnings
import nltk
from nltk import sent_tokenize
import dataclasses

max_new_tokens = 250

#nltk punkt tokenize class
nltk.download('punkt')


#model quantization using BitsandBytes Config for faster processing
@dataclasses.dataclass
class Q_values:
  Four_bit_activation : bool = True


quantization_config = BitsAndBytesConfig(
    load_in_4bit=   Q_values.Four_bit_activation,
    bnb_4bit_compute_dtype=torch.float16
)

#Img2text model
def img2txt_model(image, prompt):  
  model_id = "llava-hf/llava-1.5-7b-hf"
  pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
  prompt_instructions = "USER: <image>\n" + prompt + "\nASSISTANT:"
  outputs = pipe(image, prompt=prompt_instructions, generate_kwargs={"max_new_tokens": max_new_tokens})

  if outputs is not None and len(outputs[0]["generated_text"]) > 0:
    match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
    if match:
        # Extract the text after "ASSISTANT:"
        reply = match.group(1)
    else:
        reply = "No response found."
  else:
      reply = "No response generated."
  return reply






from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image

#model=AutoModelForCausalLM.from_pretrained("/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_use_lora/")
#model=AutoModelForCausalLM.from_pretrained("/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu05B_VL_model_lora_merge_001/")

#stage2_lm_weights=torch.load("/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_freeze_vision_eyes/")

#model.lang_model.load_state_dict(stage2_lm_weights)

#model.save_pretrained("/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_freeze_vision_eyes_no_concatination")

#No_qwen_model_name_or_path = "/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu05B_VL_model_lora_merge_001/"

#llm_model = AutoModelForCausalLM.from_pretrained( No_qwen_model_name_or_path, device_map="cuda:0")


import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

from peft import peft_model,PeftModel

from XiaYu_model001_test import processor

#raw_model_name_or_path = "/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu05B_VL_model_lora_merge_001/"
model_id = "XiaYu-VL_05B_model/model001"
tokenizer=AutoTokenizer.from_pretrained(model_id)

#processor=AutoProcessor.from_pretrained(raw_model_name_or_path)

freeze_model="/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_freeze_vision_eyes/"
#peft_model_name_or_path = "/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_use_lora/checkpoint-186050/"
#model = LlavaForConditionalGeneration.from_pretrained(freeze_model,device_map="cuda:0", torch_dtype=torch.bfloat16)
#model = PeftModel.from_pretrained(model, peft_model_name_or_path, adapter_name="peft_v1")
# trying to find the transformers/src/model/llava/modeling_llava.py, where the model is defined!
model = LlavaForConditionalGeneration.from_pretrained(
    freeze_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

#model_id = "XiaYu-VL_05B_model/model001"




processor = LlavaProcessor.from_pretrained(model_id)


prompt_text = "Could you please describe the image?\n<image>"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_file = "/home/xiongming/Documents/DATASETS/EYES/fundus_5990_split_resize/test/resized_images/2f5824d3365c.png"
#image_file = "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg"

raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors="pt").to(0, torch.float16)


output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=False))
'''
image_file = "/home/xiongming/Documents/DATASETS/EYES/fundus_5990_split_resize/test/resized_images/2f5824d3365c.png"
from PIL import Image
image = Image.open(image_file)

prompt_text = "<image>\n Can you describe the image? \n"
#inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(0, torch.float16)
inputs = processor(text=prompt_text, images=image, return_tensors="pt")

outputs=model.generate(**inputs,max_new_tokens=100)
print(processor.decode(outputs[0], skip_special_tokens=True))

'''

from sentence_transformers import SentenceTransformer

sentences=["A man is eating a burger.","A woman is singing a song."]

model_sentence=SentenceTransformer('all-MiniLM-L6-v2')

embeddings=model_sentence.encode(sentences)

from sklearn.metrics.pairwise import cosine_similarity

similarty=cosine_similarity(embeddings)

print("The semantic similarity of the image understanding is:")
print(similarty)

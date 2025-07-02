
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

modify_qwen_tokenizer_dir = "/home/xiongming/Documents/Qwen1.5-0.5B-Chat"
modify_qwen_tokenizer = AutoTokenizer.from_pretrained(modify_qwen_tokenizer_dir)

modify_qwen_tokenizer.encode("<image>")


print(modify_qwen_tokenizer.encode("<image>"))


qwen_model = AutoModelForCausalLM.from_pretrained(modify_qwen_tokenizer_dir, device_map='cuda:0',
                                                  torch_dtype=torch.bfloat16)

print(qwen_model.model.embed_tokens)

clip_model_name_or_path = (
    "/home/xiongming/Documents/PythonProject5/openai/clip-vit-large-patch14-336"
)
qwen_model_name_or_path = "/home/xiongming/Documents/Qwen1.5-0.5B-Chat"

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor

clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map="cuda:0")
llm_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name_or_path, device_map="cuda:0"
)

llm_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)
llm_tokenizer.encode("<image>")

print("llm_tonenizer.encode 的image 151646")
print(llm_tokenizer.encode("<image>")
)

from transformers import (
    LlavaForConditionalGeneration,
    LlavaConfig
)

# Initializing a CLIP-vision config
vision_config = clip_model.vision_model.config

# Initializing a Llama config
text_config = llm_model.config

# Initializing a Llava llava-1.5-7b style configuration
configuration = LlavaConfig(vision_config, text_config)

# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration(configuration)

print(model.vision_tower.vision_model.embeddings)


model.vision_tower.vision_model = clip_model.vision_model

model.language_model = llm_model

print(llm_model.model.embed_tokens.weight.data[:, :2])

print(model.language_model.model.embed_tokens.weight.data[:, :2])


print(model.config.pad_token_id)

model.config.pad_token_id = llm_tokenizer.pad_token_id
print(model.config.pad_token_id)

print(model.config.image_token_index)

print(llm_tokenizer.encode("<image>")[0])

model.config.image_token_index = llm_tokenizer.encode("<image>")[0]
print(model.config.image_token_index)

model.save_pretrained("XiaYu-VL_05B_model/model001")
llm_tokenizer.save_pretrained("XiaYu-VL_05B_model/model001")
autoprocessor = AutoProcessor.from_pretrained(clip_model_name_or_path)
autoprocessor.save_pretrained("XiaYu-VL_05B_model/model002")
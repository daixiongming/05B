from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from peft import peft_model,PeftModel
raw_model_name_or_path = "/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/XiaYu-VL_05B_model/model001/"
peft_model_name_or_path = "/home/xiongming/PycharmProjects/PythonProject1_Qwen15_05B_Chat/output_XiaYu_VL05B_model_use_lora/checkpoint-186050/"
model = LlavaForConditionalGeneration.from_pretrained(raw_model_name_or_path,device_map="cuda:0", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, peft_model_name_or_path, adapter_name="peft_v1")
processor = AutoProcessor.from_pretrained(raw_model_name_or_path)
model.eval()
print('ok')

#https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/34
#processor.patch_size = model.config.vision_config.patch_size

#processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy



from train_llava.data import LlavaDataset

llavadataset = LlavaDataset("/home/xiongming/Documents/PythonProject5/liuhaotian/LLaVA-CC3M-Pretrain-595K")
print(len(llavadataset))
print(llavadataset[10])


testdata = llavadataset[1303]
print(testdata)

from PIL import Image
Image.open(testdata[2])


def build_model_input(model, processor, testdata: tuple):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": testdata[0]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(prompt)
    # print("*"*20)
    image = Image.open(testdata[2])
    inputs = processor(text=prompt, images=image, return_tensors="pt")




    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=20)

    generate_ids = [
        oid[len(iids):] for oid, iids in zip(generate_ids, inputs.input_ids)
    ]


    gen_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return gen_text


build_model_input(model, processor, testdata)


model = model.merge_and_unload()
model.save_pretrained("output_XiaYu05B_VL_model_lora_merge_001")
processor.save_pretrained("output_XiaYu05B_VL_model_lora_merge_001")





import torch
# 改动：从 transformers 导入，不再使用 modelscope
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ========== 本地模型路径 ==========
model_path = "/home/ps/LLM/CMG-VLM"
# ==================================

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", # flash_attn报错就删掉这行
    device_map="auto",
    trust_remote_code=True, # 5.x版本建议加上，兼容Lingshu模型配置
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/ps/data/guzhao/datasets/EX-Datasets/SLAKE/data/source_xmlab132.jpg",
            },
            {"type": "text", "text": "请描述这张医学影像，给出诊断提示。"},
        ],
    }
]

# 构造prompt模板
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# 推理生成
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.8
    )
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("模型输出：")
print(output_text[0])


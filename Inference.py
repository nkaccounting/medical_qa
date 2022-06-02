import time

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

model_dir = "./medical-clm/checkpoint-10000"

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

if torch.cuda.is_available():
    text_generator = TextGenerationPipeline(model, tokenizer, device=0)
else:
    text_generator = TextGenerationPipeline(model, tokenizer)

while True:
    text = input('请输入您想咨询的疾病问题，目前仅支持（儿科，妇产科，男科，内科，外科，肿瘤科）:')

    text = '<QBOS>{text}<QEOS>'.format(text=text)

    # n = len(text)
    t1 = time.time()

    res = text_generator(
        text,
        max_length=300,
        do_sample=True,
        num_return_sequences=5,
        top=0.5,
        eos_token_id=4,
        pad_token_id=4,
        return_full_text=False,  # 不返回全部的文本，只返回生成的部分
        clean_up_tokenization_spaces=True  # 删除生成文本当中的空格
    )
    t2 = time.time()
    print("生成5个答案所用时间，", t2 - t1)

    for i, r in enumerate(res):
        print('候选回答{i}:'.format(i=i), r['generated_text'])

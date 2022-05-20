from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

model_dir = "./medical-clm"

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
text_generator = TextGenerationPipeline(model, tokenizer)

while True:
    text = input('请输入您想咨询的疾病问题，目前仅支持（儿科，妇产科，男科，内科，外科，肿瘤科）:')

    text = '<BOS>{text}<EOS>'.format(text=text)

    n = len(text)

    res = text_generator(
        text,
        max_length=300,
        do_sample=True,
        num_return_sequences=5,
        top=0.5,
        eos_token_id=2,
        pad_token_id=2
    )

    for i, r in enumerate(res):
        print('候选回答{i}:'.format(i=i), r['generated_text'][n + 1:])

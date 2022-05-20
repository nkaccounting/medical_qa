from transformers import BertTokenizer

tokenizers_my = BertTokenizer.from_pretrained('../gpt2-chinese-cluecorpussmall')

special_tokens_dict = {"bos_token": "<BOS>", "eos_token": "<EOS>"}

num_added_toks = tokenizers_my.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")


special_tokens_dict = {"additional_special_tokens": ["<QOS>", "<WOS>"]}

num_added_toks = tokenizers_my.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")

assert tokenizers_my.bos_token == "<BOS>"

assert tokenizers_my.eos_token == "<EOS>"

# 添加一般的special token


print(tokenizers_my.SPECIAL_TOKENS_ATTRIBUTES)

a = tokenizers_my.encode('<BOS>希望没事<EOS><EOS><QOS><WOS>')
print(a)
# [101, 1, 2361, 3307, 3766, 752, 2, 2, 102]


a = tokenizers_my.tokenize('<BOS>希望没事<EOS><EOS>')
print(a)
# before
# ['<', '[UNK]', '>', '希', '望', '没', '事', '<', '[UNK]', '>', '<', '[UNK]', '>']
# after
# ['<BOS>', '希', '望', '没', '事', '<EOS>', '<EOS>']


# tokenizers_my.save_pretrained('./my_token')


# [PAD]
# <BOS>
# <EOS>
# [unused3]
# [unused4]
# [unused5]
# [unused6]
# [unused7]

# 要点，修改vocabulary，和special_map两个文件

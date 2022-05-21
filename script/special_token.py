from transformers import BertTokenizer

tokenizers_my = BertTokenizer.from_pretrained('../gpt2-chinese-cluecorpussmall')

# special_tokens_dict = {"bos_token": "<BOS>", "eos_token": "<EOS>"}
#
# num_added_toks = tokenizers_my.add_special_tokens(special_tokens_dict)
# print("We have added", num_added_toks, "tokens")

# assert tokenizers_my.bos_token == "<BOS>"
#
# assert tokenizers_my.eos_token == "<EOS>"


special_tokens_dict = {"additional_special_tokens": ["<QBOS>", "<QEOS>", "<ABOS>", "<AEOS>"]}

num_added_toks = tokenizers_my.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")

# 添加一般的special token


print(tokenizers_my.SPECIAL_TOKENS_ATTRIBUTES)

a = tokenizers_my.encode('<QBOS>癫痫病人请假吗？<QEOS><ABOS>病情分析：你好！你这个可以去医院看看指导意见：可以做一些检查，如果是癫痫，医生应该会给你开的。，癫痫病患者在及时治疗之外，患者在生活中还需要注意要保持良好的心情，好的心情对疾病的恢复很有帮助，希望上述的答案可以帮助到你，谢谢！<AEOS>')
print(a)
# [101, 1, 2361, 3307, 3766, 752, 2, 2, 102]


a = tokenizers_my.tokenize('<QBOS>癫痫病人请假吗？<QEOS><ABOS>病情分析：你好！你这个可以去医院看看指导意见：可以做一些检查，如果是癫痫，医生应该会给你开的。，癫痫病患者在及时治疗之外，患者在生活中还需要注意要保持良好的心情，好的心情对疾病的恢复很有帮助，希望上述的答案可以帮助到你，谢谢！<AEOS>')
print(a)
# before
# ['<', '[UNK]', '>', '希', '望', '没', '事', '<', '[UNK]', '>', '<', '[UNK]', '>']
# after
# ['<BOS>', '希', '望', '没', '事', '<EOS>', '<EOS>']


tokenizers_my.save_pretrained('./my_token')


# [PAD]
# <BOS>
# <EOS>
# [unused3]
# [unused4]
# [unused5]
# [unused6]
# [unused7]

# 要点，修改vocabulary，和special_map两个文件

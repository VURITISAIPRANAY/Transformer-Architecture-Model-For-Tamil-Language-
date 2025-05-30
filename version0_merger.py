# main reference https://gucci-j.github.io/post/en/vocab-expansion/
# this doesn't add tokens to the 'added_tokens' in the tokenizer.json,
# this adds the vocab perfectly, in order and without adding the common ones.
# but this doesn't add the 'merges' in order.


from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sentencepiece as sp
import math
import copy

from tokenizers.models import BPE



#main llama
ll_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
ll_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
vocab = ll_tokenizer.get_vocab()
tokenizer_json = json.loads(ll_tokenizer._tokenizer.to_str())
print(tokenizer_json["pre_tokenizer"])
# print(tokenizer_json["model"]["merges"])
merges = tokenizer_json["model"]["merges"]
print(merges[:10])

#auxilary
sar_model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")
sar_tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
sar_tokenizer_json = json.loads(sar_tokenizer._tokenizer.to_str())
sar_merges = sar_tokenizer_json["model"]["merges"]
print(sar_merges[:10])

tot_merges=len(sar_merges)
progress_percent_step = 1
progress = progress_percent_step

# merge the tokenizers
num_new_token = 0
max_new_token = 90000

ret_vocab = copy.copy(vocab)
ret_merges = []
old_merges = copy.deepcopy(merges)
for index,merge in enumerate(sar_merges):
    # vocab
    token_1, token_2 = merge[0],merge[1]
    token = token_1 + token_2
    if num_new_token < max_new_token:
        if token_1 not in ret_vocab and token_2 not in ret_vocab: # both are new
            ret_vocab[token_1] = len(vocab) + num_new_token
            ret_vocab[token_2] = len(vocab) + num_new_token + 1
            num_new_token += 2
        elif token_1 not in ret_vocab and token_2 in ret_vocab: # new + existing
            ret_vocab[token_1] = len(vocab) + num_new_token
            num_new_token += 1
        elif token_1 in ret_vocab and token_2 not in ret_vocab: # old + existing
            ret_vocab[token_2] = len(vocab) + num_new_token
            num_new_token += 1
        else: # both are existing tokens
            pass
        if token not in ret_vocab:
            ret_vocab[token] = len(vocab) + num_new_token
            num_new_token += 1
    # merge
    merge_str=" ".join(merge)
    if merge_str in [" ".join(old_merge) for old_merge in old_merges]:
        old_merges.remove(merge)
        ret_merges.append(merge)
    elif token in ret_vocab and token_1 in ret_vocab and token_2 in ret_vocab:
        ret_merges.append(merge)

    progress_percent=((index+1)/tot_merges) *100
    if progress_percent >= progress:
        print(f"progress- {math.floor(progress_percent)}")
        progress += progress_percent_step

print(ret_merges[:10])
print(old_merges[:10])

merges = ret_merges + old_merges
vocab = ret_vocab
ll_tokenizer.backend_tokenizer.model = BPE(
    vocab=vocab,
    merges=[(merge[0], merge[1]) for merge in merges],
    fuse_unk=False,
)





# inputs = tokenizer(sequ, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=5)
# result = tokenizer.decode(outputs[0])



ll_tokenizer.save_pretrained("/Users/pranay/PycharmProjects/PythonProject/tamil_fast_tokenizer",legacy_format=False,filename_prefix='Fast_Tamil_')


sequ="தமிழ் அரிச்சுவடி (Tamil script) என்பது தமிழ் மொழியில் உள்ள எழுத்துகளின் வரிசை ஆகும். அரி என்னும் முன்னடை சிறு என்னும் பொருள் கொண்டது. இவை தமிழ் அகரவரிசை, தமிழ் நெடுங்கணக்கு போன்ற சொற்களாலும் குறிப்பிடப்படுகின்றன. தமிழில் 12 உயிரெழுத்துகளும் 18 மெய்யெழுத்துகளும் ஓர் ஆய்த எழுத்தும் 216 உயிர்மெய் எழுத்துகளுமாக மொத்தம் 247 எழுத்துகள், தமிழ் நெடுங்கணக்கில் உள்ளன. தற்காலத்தில் வழங்கும் கிரந்த எழுத்துகள் தமிழ் நெடுங்கணக்கைச் சேர்ந்தவையல்ல."
tokens=sar_tokenizer.tokenize(sequ)
print(tokens)
ids=sar_tokenizer.convert_tokens_to_ids(tokens)
print(len(ids))
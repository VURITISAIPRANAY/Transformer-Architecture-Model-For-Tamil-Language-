# this code, adds only unique tokens to the 'added_tokens', and it shuffles the tokens,
# so the token id's are not in order (they are not starting from 1 and going in an increasing order),
# and also this merges the common 'merges' from both the tokenizer.json files.
# it doesn't add the unique ones only. but this 'merger' has order.


from transformers import AutoTokenizer
import json
import os
import shutil
import copy


# Replace with the actual names of the tokenizers on Hugging Face
llama3_tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
other_tokenizer_name = "sarvamai/sarvam-1"
output_tokenizer_path = "/home/pranay/PycharmProjects/TAMILgpt/combined_tokenizer"
temp_tokenizer_path = "/home/pranay/PycharmProjects/TAMILgpt/temp"

# Load the LLaMA 3 tokenizer
llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_tokenizer_name)

# Load the other tokenizer
other_tokenizer = AutoTokenizer.from_pretrained(other_tokenizer_name)

# Get the vocabularies
llama3_vocab = llama3_tokenizer.get_vocab()
other_vocab = other_tokenizer.get_vocab()


# Identify new tokens from the other tokenizer
new_tokens = {}
for token, id in other_vocab.items():
    if token not in llama3_vocab:
        new_tokens[token] = len(llama3_vocab) + len(new_tokens)

# Add the new tokens to the LLaMA 3 tokenizer
llama3_tokenizer.add_tokens(list(new_tokens.keys()))

# Save the (initially modified) LLaMA 3 tokenizer configuration
llama3_tokenizer.save_pretrained(output_tokenizer_path)

# --- Manual Manipulation of tokenizer.json ---

output_tokenizer_config_path = os.path.join(output_tokenizer_path, "tokenizer.json")

# Save the other tokenizer temporarily
other_tokenizer.save_pretrained(temp_tokenizer_path)
other_tokenizer_config_path = os.path.join(temp_tokenizer_path, "tokenizer.json")

# Load the tokenizer.json of the other tokenizer to get its merges
with open(other_tokenizer_config_path, 'r', encoding='utf-8') as f:
    other_tokenizer_data = json.load(f)

# Load the tokenizer.json of the (modified) LLaMA 3 tokenizer
with open(output_tokenizer_config_path, 'r', encoding='utf-8') as f:
    combined_tokenizer_data = json.load(f)

# performing a sorting because the get_vocab() is returning a dict in shuffeled order. and that's not acceptable when using the merges.
ll_dict=llama3_tokenizer.get_vocab()
sorted_ll_dict=dict(sorted(ll_dict.items(), key=lambda x:x[1]))
print(sorted_ll_dict)

# Update the vocabulary in the combined tokenizer data

combined_tokenizer_data['model']['vocab'] = llama3_tokenizer.get_vocab()

# Get the merges from both tokenizers

if 'model' in combined_tokenizer_data and 'merges' in combined_tokenizer_data['model'] and \
   'model' in other_tokenizer_data and 'merges' in other_tokenizer_data['model']:
    llama3_merges = combined_tokenizer_data['model']['merges']
    other_merges = other_tokenizer_data['model']['merges']

    # Append all merges from the other tokenizer
    combined_merges = llama3_merges + other_merges
    combined_tokenizer_data['model']['merges'] = combined_merges

    # Update 'ignore_merges' in the combined tokenizer to False
    if 'ignore_merges' in combined_tokenizer_data['model']:
        combined_tokenizer_data['model']['ignore_merges'] = False
else:
    print("Could not find 'merges' in one or both tokenizer.json files.")

# Save the final combined tokenizer.json
with open(output_tokenizer_config_path, 'w', encoding='utf-8') as f:
    json.dump(combined_tokenizer_data, f, indent=2, ensure_ascii=False)

# Clean up the temporary directory
shutil.rmtree(temp_tokenizer_path, ignore_errors=True)

print(f"Combined tokenizer configuration saved to {output_tokenizer_path}")
print("Warning: We have appended all merges from the other tokenizer. The order of these appended merges relative to the original LLaMA 3 merges might not be optimal and could impact performance. Manual review and potential reordering of the 'merges' list in tokenizer.json might be necessary for the best results.")
print("Consider reviewing other configuration parameters like 'unk_token', 'fuse_unk', and 'byte_fallback' in the combined tokenizer.json to see if they should be aligned with either of the original tokenizers based on your requirements.")
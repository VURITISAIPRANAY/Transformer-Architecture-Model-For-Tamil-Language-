# this code is to check the entire pipeline from normalizer to pretokenizer(split and bytelevel)


from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from tokenizers import Regex

# --- Configuration ---
# Step 1: Normalizer - Replace spaces with 'Ġ'
normalizer = Replace(pattern=" ", content="Ġ")

# Step 2: Pre-tokenizer sequence
# First pre-tokenizer: Split with regex pattern
split_pattern = Regex(r"(?=\u0120)|(?<=[.,!?;:])(?=.)|(?<=.)(?=[.,!?;:])")
split_pretok = Split(pattern=split_pattern, behavior="isolated", invert=False)

# Second pre-tokenizer: ByteLevel
byte_level_pretok = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)

# Combine pre-tokenizers into a sequence
pretok_sequence = Sequence([split_pretok, byte_level_pretok])

# --- Input Text Examples ---
example_texts = [
    "Hello world! This is a test.",
    "Multiple  spaces between. Leading spaces work too.",
    "சென்னை: சென்னை சேர்ந்த வழக்கறிஞர்"
]

# --- Process and Display ---
for i, text in enumerate(example_texts):
    print("\n" + "=" * 60)
    print(f"EXAMPLE {i + 1}: '{text}'")
    print("=" * 60)

    # Step 1: Apply normalizer
    normalized_text = normalizer.normalize_str(text)
    print("\nAfter Normalizer (Replace spaces with 'Ġ'):")
    print(f"'{normalized_text}'")

    # Step 2a: Apply first pre-tokenizer (Split)
    split_output = split_pretok.pre_tokenize_str(normalized_text)
    split_tokens = [token for token, _ in split_output]
    print("\nAfter Split Pre-tokenizer:")
    print(split_tokens)

    # Step 2b: Apply the full pre-tokenizer sequence
    # We need to transform the output format from the first pre-tokenizer
    # to match what the sequence expects
    sequence_output = []
    sequence_output = pretok_sequence.pre_tokenize_str(normalized_text)

    # Extract tokens from the final pre-tokenized output
    final_tokens = [token for token, _ in sequence_output]
    print("\nAfter ByteLevel Pre-tokenizer (Final Output):")
    print(final_tokens)
    print("-" * 60)


# # output i got
# ============================================================
# EXAMPLE 1: 'Hello world! This is a test.'
# ============================================================
#
# After Normalizer (Replace spaces with 'Ġ'):
# 'HelloĠworld!ĠThisĠisĠaĠtest.'
#
# After Split Pre-tokenizer:
# ['Hello', 'Ġworld', '!', 'ĠThis', 'Ġis', 'Ġa', 'Ġtest', '.']
#
# After ByteLevel Pre-tokenizer (Final Output):
# ['Hello', 'Äłworld', '!', 'ÄłThis', 'Äłis', 'Äła', 'Äłtest', '.']

# after checking the output and the documentation of bytelevel, i came to a conclusion to remome that step, because it bad at converting the Indian languages. it is a significant step for BPE type tokenizers (to handle typos and unk tokens well.) but UTF-8 is not the right one for Indian Languages, especially because i borrowed the merge rules from sarvam-1.



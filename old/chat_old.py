# FILES IN 'old' ARE DEPRECATED EXAMPLES OF INFERENCE, USE 'demo.py' AS NEW EXAMPLE

import torch
from sae import SparseAutoencoder
from mistral7b import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from utils.generate import generate


CLAMPED_FEATURES = [
    {"feat_index": 79557, "val": 110.0}  # EXAMPLE: Pacific Ocean feature
]


D_MODEL = 4096
D_HIDDEN = 131072
MISTRAL_MODEL_PATH = "PATH TO MISTRAL7b WEIGHTS"
SAE_MODEL_PATH = "PATH TO MODEL WEIGHTS"

print("Loading Model...")

model = SparseAutoencoder(D_MODEL, D_HIDDEN)
model.load_state_dict(torch.load(SAE_MODEL_PATH))
mistral7b = Transformer.from_folder(MISTRAL_MODEL_PATH)
tokenizer = MistralTokenizer.from_file(f"{MISTRAL_MODEL_PATH}/tokenizer.model.v3")

model = model.to("cuda")
model = model.eval()
mistral7b = mistral7b.to("cuda")

print("Model Loaded!\n")

messages = []

while True:
    prompt = input(f"\033[34mUser\033[0m: ")
    messages += [UserMessage(content=prompt)]
    completion_request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens = generate(
        [tokens],
        mistral7b,
        max_tokens=128,
        temperature=0.3,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        sae=model,
        features=CLAMPED_FEATURES,
    )

    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])[0]
    messages += [AssistantMessage(content=result)]
    print(f"\033[31mModel\033[0m: {result}")

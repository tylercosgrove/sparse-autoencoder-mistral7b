import torch
from sae import SparseAutoencoder
from mistral7b import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from utils.generate import generate

D_MODEL = 4096
D_HIDDEN = 131072
MISTRAL_MODEL_PATH = "PATH TO MISTRAL7b WEIGHTS"
SAE_MODEL_PATH = "PATH TO MODEL WEIGHTS"

model = SparseAutoencoder(D_MODEL, D_HIDDEN)
model.load_state_dict(torch.load(SAE_MODEL_PATH))
mistral7b = Transformer.from_folder(MISTRAL_MODEL_PATH)
tokenizer = MistralTokenizer.from_file(f"{MISTRAL_MODEL_PATH}/tokenizer.model.v3")

mistral7b = mistral7b.to("cuda")
model = model.to("cuda")
model = model.eval()


prompt = "Hello there!"
completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens = generate(
    [tokens],
    mistral7b,
    max_tokens=128,
    temperature=0.0,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    sae=model,
)

result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

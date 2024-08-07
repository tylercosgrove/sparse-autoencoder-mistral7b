from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

# CODE IS BASED ON: https://github.com/jbloomAus/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb

device = "cpu"
SAE_PATH = "path to folder containing 'cfg.json' and 'sae_weights.safetensors' from HuggingFace"
FEATURE_INDEX = 79557 # pacific ocean feature
STEERING_ON = True


model = HookedTransformer.from_pretrained("mistral-7b-instruct", dtype="float16", device=device)
sae = SAE.load_from_pretrained(SAE_PATH, dtype="float16", device=device)

steering_vector = sae.W_dec[FEATURE_INDEX]
example_prompt = "Write me a poem."
coeff = 500
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)


def steering_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return
    
    if STEERING_ON:
        resid_pre += coeff * steering_vector


def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    return result

def run_generate(example_prompt):
    model.reset_hooks()
    editing_hooks = [(sae.cfg.hook_name, steering_hook)]
    res = hooked_generate([example_prompt], editing_hooks, seed=None, **sampling_kwargs)

    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

run_generate(example_prompt)
# Sparse Autoencoder for Mistral 7b

Read my blog post about this [here](https://www.tylercosgrove.com/blog/exploring-sae/).

This a sparse autoencoder I trained on the residual activations of Mistral 7b. The implementation is largely based on [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (Templeton, et al. 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [Scaling and evaluating sparse autoencoders (Gao et al. 2024)](https://arxiv.org/abs/2406.04093v1).

I was inspired to do this after seeing [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude).

HuggingFace link for the model weights: [https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16](https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16)

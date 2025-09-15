#console script to compare FP32 vs INT8 embeddings (cosine similarity + inference time).

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from quantmodel import QuantizedSentenceEncoder


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_original(texts, model_name="deepvk/USER-BGE-M3", normalize=True, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


if __name__ == "__main__":
    texts = ["Привет мир!", "Hello world!"]

    orig_embeddings = encode_original(texts)
    quant_model = QuantizedSentenceEncoder()
    quant_embeddings = quant_model.encode(texts)

    print("Original shape:", orig_embeddings.shape)
    print("Quantized shape:", quant_embeddings.shape)

    cos_sim = np.dot(orig_embeddings[0], quant_embeddings[0])
    print(f"Cosine similarity between original & quantized: {cos_sim:.4f}")

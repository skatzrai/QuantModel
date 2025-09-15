#code for interactive demo with Streamlit.

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from quantmodel import QuantizedSentenceEncoder


# ======================
# Original model
# ======================
@st.cache_resource
def load_original_model():
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")
    model = AutoModel.from_pretrained("deepvk/USER-BGE-M3")
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_original(texts, normalize=True, device="cpu"):
    tokenizer, model = load_original_model()
    model = model.to(device)

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


# ======================
# Quantized model
# ======================
@st.cache_resource
def load_quantized_model():
    return QuantizedSentenceEncoder()


# ======================
# Streamlit UI
# ======================
st.title("Embeddings comparison: Original vs Quantized (ONNX)")

text1 = st.text_input("Enter the first text", "Привет мир!")
text2 = st.text_input("Enter the second text", "Hello world!")

if st.button("Compare"):
    texts = [text1, text2]

    st.write("⚡ Processing...")

    # Original
    orig_emb = encode_original(texts)
    # Quantized
    quant_model = load_quantized_model()
    quant_emb = quant_model.encode(texts)

    # Cosine similarity within each model
    cos_orig = float(np.dot(orig_emb[0], orig_emb[1]))
    cos_quant = float(np.dot(quant_emb[0], quant_emb[1]))

    # Cosine similarity between original and quantized
    cos_cross = float(np.dot(orig_emb[0], quant_emb[0]))

    st.subheader("Results")
    st.write(f"Original cosine(text1, text2): **{cos_orig:.4f}**")
    st.write(f"Quantized cosine(text1, text2): **{cos_quant:.4f}**")
    st.write(f"Cosine(original text1 vs quantized text1): **{cos_cross:.4f}**")

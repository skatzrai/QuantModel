# quantmodel.py

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


class QuantizedSentenceEncoder:
    def __init__(self, model_name: str = "skatzR/USER-BGE-M3-ONNX-INT8", device: str = None):
        """
        Universal loader for quantized ONNX model.
        :param model_name: HuggingFace repo id
        :param device: "cpu" or "cuda". If None, selected automatically.
        """

        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine device
        if device is None:
            self.device = "cuda" if ort.get_device() == "GPU" else "cpu"
        else:
            self.device = device

        # Download ONNX file from HuggingFace
        model_path = hf_hub_download(repo_id=model_name, filename="model_quantized.onnx")

        # Initialize ONNX runtime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling (as in the original)
        model_output: np.array (batch_size, seq_len, hidden_size)
        attention_mask: torch.Tensor (batch_size, seq_len)
        """
        token_embeddings = model_output  # (bs, seq_len, hidden_size)
        input_mask_expanded = np.expand_dims(attention_mask.numpy(), -1)  # (bs, seq_len, 1)

        # Mask and average
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def encode(self, texts, normalize: bool = True, batch_size: int = 32):
        """
        Get sentence embeddings
        :param texts: str или list[str]
        :param normalize: whether to apply L2 normalization
        :param batch_size: batch size
        :return: np.array (num_texts, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # Remove unnecessary tokens (e.g., token_type_ids)
            ort_inputs = {k: v.cpu().numpy() for k, v in enc.items() if k in self.input_names}

            # Forward pass through ONNX
            ort_outs = self.session.run([self.output_name], ort_inputs)
            token_embeddings = ort_outs[0]  # (bs, seq_len, hidden_size)

            # Pooling
            embeddings = self._mean_pooling(token_embeddings, enc["attention_mask"])

            # Normalization
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

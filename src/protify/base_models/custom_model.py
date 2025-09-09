from transformers import AutoModel


"""
Custom models are currently supposed to load completely from AutoModel.from_pretrained(path, trust_remote_code=True)
They should return embeddings from their forward pass.
"""


def build_custom_model(model_path: str):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


# py -m src.protify.base_models.custom_model
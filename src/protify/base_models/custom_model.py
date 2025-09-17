from transformers import AutoModel, AutoTokenizer


"""
Custom models are currently supposed to load completely from AutoModel.from_pretrained(path, trust_remote_code=True)
They should return embeddings from their forward pass.
"""


def build_custom_model(model_path: str):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


def build_custom_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.custom_model
    model, tokenizer = build_custom_model('lhalle/esm2_t6_8M_UR50D')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))

from pero_pretraining.models.autoencoders import VGGEncoder, VGGDecoder, AE, VQVAE

def init_model(model_definition):
    model_type = model_definition.get("type", "ae")

    encoder = VGGEncoder()
    decoder = VGGDecoder()

    if model_type == "ae":
        model = AE(encoder, decoder)
    elif model_type == "vqvae":
        num_embeddings = model_definition.get("num_embeddings", 1024)
        embeddings_dim = model_definition.get("embeddings_dim", 512)

        model = VQVAE(encoder, decoder, num_embeddings=num_embeddings, embeddings_dim=embeddings_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


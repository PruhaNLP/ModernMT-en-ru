from transformers import PretrainedConfig


class EncoderDecoderConfig(PretrainedConfig):
    model_type = "modernbert_enc_dec"

    def __init__(
        self,
        vocab_size: int = 50370,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        intermediate_size: int = 576,
        decoder_num_layers: int = 6,
        pad_token_id: int = 50283,
        decoder_start_token_id: int = 50369,
        eos_token_id: int = 50368,
        bos_token_id: int = 50369,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 160000.0,
        tie_word_embeddings: bool = True,
        encoder_config: dict = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.decoder_num_layers = decoder_num_layers
        self.num_hidden_layers = decoder_num_layers
        self.decoder_start_token_id = decoder_start_token_id
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.encoder_config = encoder_config
        self.is_encoder_decoder = True

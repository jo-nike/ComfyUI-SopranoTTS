import torch
from soprano import SopranoTTS as SopranoModel


class SopranoLoader:
    """Loads the SopranoTTS model for reuse across multiple generations."""

    _model = None
    _model_config = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["transformers", "lmdeploy", "auto"], {"default": "transformers"}),
                "cache_size_mb": ("INT", {"default": 10, "min": 1, "max": 100}),
                "decoder_batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("SOPRANO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/tts"

    def load_model(self, backend, cache_size_mb, decoder_batch_size):
        config = (backend, cache_size_mb, decoder_batch_size)

        if SopranoLoader._model is None or SopranoLoader._model_config != config:
            print(f"[SopranoTTS] Loading model with backend={backend}...")
            SopranoLoader._model = SopranoModel(
                backend=backend,
                device='cuda',
                cache_size_mb=cache_size_mb,
                decoder_batch_size=decoder_batch_size
            )
            SopranoLoader._model_config = config
            print("[SopranoTTS] Model loaded successfully.")
        else:
            print("[SopranoTTS] Using cached model.")

        return (SopranoLoader._model,)


class SopranoTTS:
    """Generate speech from text using SopranoTTS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SOPRANO_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of Soprano text to speech."}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/tts"

    def generate(self, model, text, temperature, top_p, repetition_penalty):
        if not text.strip():
            # Return silence if empty text
            waveform = torch.zeros((1, 1, 32000), dtype=torch.float32)
            return ({"waveform": waveform, "sample_rate": 32000},)

        audio_tensor = model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        # Convert from (samples,) to (batch, channels, samples)
        waveform = audio_tensor.unsqueeze(0).unsqueeze(0).float()

        return ({"waveform": waveform, "sample_rate": 32000},)


class SopranoTTSBatch:
    """Generate speech from multiple texts using SopranoTTS batch processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SOPRANO_MODEL",),
                "texts": ("STRING", {"multiline": True, "default": "First sentence to speak.\nSecond sentence to speak."}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/tts"

    def generate(self, model, texts, temperature, top_p, repetition_penalty):
        # Split by newlines and filter empty lines
        text_list = [t.strip() for t in texts.split('\n') if t.strip()]

        if not text_list:
            # Return silence if no valid text
            waveform = torch.zeros((1, 1, 32000), dtype=torch.float32)
            return ({"waveform": waveform, "sample_rate": 32000},)

        audio_tensors = model.infer_batch(
            text_list,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        # Concatenate all audio tensors
        combined = torch.cat(audio_tensors, dim=0)

        # Convert from (samples,) to (batch, channels, samples)
        waveform = combined.unsqueeze(0).unsqueeze(0).float()

        return ({"waveform": waveform, "sample_rate": 32000},)


class SopranoTTSStream:
    """Generate speech using streaming mode (lmdeploy backend only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SOPRANO_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a streaming test."}),
                "chunk_size": ("INT", {"default": 1, "min": 1, "max": 10}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/tts"

    def generate(self, model, text, chunk_size, temperature, top_p, repetition_penalty):
        if not text.strip():
            waveform = torch.zeros((1, 1, 32000), dtype=torch.float32)
            return ({"waveform": waveform, "sample_rate": 32000},)

        try:
            chunks = []
            for chunk in model.infer_stream(
                text,
                chunk_size=chunk_size,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            ):
                chunks.append(chunk)

            if chunks:
                combined = torch.cat(chunks, dim=0)
            else:
                combined = torch.zeros(32000, dtype=torch.float32)

        except NotImplementedError:
            raise RuntimeError(
                "Streaming is only supported with the 'lmdeploy' backend. "
                "The 'transformers' backend does not support streaming. "
                "Use the regular SopranoTTS node instead."
            )

        # Convert from (samples,) to (batch, channels, samples)
        waveform = combined.unsqueeze(0).unsqueeze(0).float()

        return ({"waveform": waveform, "sample_rate": 32000},)

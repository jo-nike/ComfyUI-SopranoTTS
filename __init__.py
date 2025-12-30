from .nodes import SopranoLoader, SopranoTTS, SopranoTTSBatch, SopranoTTSStream

NODE_CLASS_MAPPINGS = {
    "SopranoLoader": SopranoLoader,
    "SopranoTTS": SopranoTTS,
    "SopranoTTSBatch": SopranoTTSBatch,
    "SopranoTTSStream": SopranoTTSStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SopranoLoader": "Soprano TTS Loader",
    "SopranoTTS": "Soprano TTS",
    "SopranoTTSBatch": "Soprano TTS Batch",
    "SopranoTTSStream": "Soprano TTS Stream",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

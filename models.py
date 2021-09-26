from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sentence_transformers import SentenceTransformer

def load_wav_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    return model, processor

def load_sentence_model():
    return SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
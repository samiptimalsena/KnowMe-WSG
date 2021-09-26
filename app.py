import gradio as gr
import numpy as np
from models import load_sentence_model, load_wav_model
from utils import QA, parse_wav_transcription
from sentence_transformers.util import cos_sim

wav_model, wav_processor = load_wav_model()
sentence_model = load_sentence_model()
encoded_sentence = np.load('embeddings.npy')

def knowMe(audio):
    recorded_sentence = parse_wav_transcription(audio.name, wav_model, wav_processor)
    recorded_enc = sentence_model.encode(recorded_sentence)

    cosine_sim = [cos_sim(recorded_enc, context_enc) for context_enc in encoded_sentence]
    idx = np.argmax(cosine_sim)
    return QA[idx]

iface = gr.Interface(fn=knowMe, 
        inputs = gr.inputs.Audio(source="microphone", type="file", label="Audio"),
        outputs=gr.outputs.Textbox(),
        title="Know Me - WSG",
        description="Ask to Know",
        allow_flagging=False
)
iface.launch()
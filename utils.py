import torch
import librosa

QA = {
    0: "My brother name is Sanup.",
    1: "I live in Kathmandu.",
    2: "I like programming and playing guitar.",
    3: "I have 4 members in my family.",
    4: "I am currently pursuing my bachelors in Computer engineering in Kathmandu University."
}

context_sentence = ["What is your brother name?",
                     "Where do you live?",
                      "What is your hobby?",
                       "How many members are there in your family?",
                        "What are you studying?"]

def parse_wav_transcription(wav_file, model, processor):
    audio_input, sample_rate = librosa.load(wav_file, sr=16000)
    input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription
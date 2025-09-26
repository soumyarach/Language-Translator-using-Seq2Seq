import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate

# Sample data
data = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("i love you", "je t aime"),
    ("see you soon", "à bientôt"),
    ("how are you", "comment allez vous"),
]

eng_texts, fr_texts = zip(*data)

def build_vocab(sentences):
    words = set()
    for s in sentences:
        words.update(s.lower().split())
    word2idx = {w: i+1 for i, w in enumerate(sorted(words))}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

eng_word2idx, eng_idx2word = build_vocab(eng_texts)
fr_word2idx, fr_idx2word = build_vocab(fr_texts)

input_vocab_size = len(eng_word2idx) + 1
target_vocab_size = len(fr_word2idx) + 1

def to_seq(sentences, word2idx, max_len=None):
    seqs = []
    for s in sentences:
        seqs.append([word2idx.get(w, 0) for w in s.lower().split()])
    max_len = max_len or max(len(s) for s in seqs)
    padded = np.zeros((len(seqs), max_len), dtype="int32")
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, max_len

eng_tensor, max_eng_len = to_seq(eng_texts, eng_word2idx)
fr_tensor, max_fr_len = to_seq(fr_texts, fr_word2idx)

latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(input_vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
enc_outputs, state_h, state_c = encoder_lstm(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(target_vocab_size, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

attn = Attention()
attn_out = attn([dec_outputs, enc_outputs])
concat = Concatenate(axis=-1)([dec_outputs, attn_out])

decoder_dense = Dense(target_vocab_size, activation="softmax")
final_outputs = decoder_dense(concat)

model = Model([encoder_inputs, decoder_inputs], final_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Prepare decoder data
decoder_in = fr_tensor[:, :-1]
decoder_out = fr_tensor[:, 1:]

print("🔹 Training model...")
model.fit(
    [eng_tensor, decoder_in],
    np.expand_dims(decoder_out, -1),
    epochs=30,
    batch_size=2,
    verbose=0,
)
print("Training completed!")

# Encoder model for inference
encoder_model = Model(encoder_inputs, [enc_outputs, state_h, state_c])

# Decoder inputs for inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
encoder_outputs_input = Input(shape=(None, latent_dim))

decoder_inputs_single = Input(shape=(1,))
dec_emb_single = dec_emb_layer(decoder_inputs_single)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb_single, initial_state=[decoder_state_input_h, decoder_state_input_c]
)
attn_out2 = attn([decoder_outputs2, encoder_outputs_input])
concat2 = Concatenate(axis=-1)([decoder_outputs2, attn_out2])
decoder_outputs2 = decoder_dense(concat2)

decoder_model = Model(
    [decoder_inputs_single, encoder_outputs_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2, state_h2, state_c2],
)

translation_history = []

def translate(sentence):
    seq, _ = to_seq([sentence], eng_word2idx, max_len=max_eng_len)
    enc_outs, h, c = encoder_model.predict(seq)

    target_seq = np.array([[0]])  # start token assumed as 0 index
    decoded = []

    for _ in range(max_fr_len):
        preds, h, c = decoder_model.predict([target_seq, enc_outs, h, c])
        word_id = np.argmax(preds[0, -1, :])
        if word_id == 0:  # stop if zero/padding token
            break
        decoded.append(fr_idx2word.get(word_id, ""))
        target_seq = np.array([[word_id]])

    result = " ".join(decoded)
    translation_history.append((sentence, result))
    return result

print("Translate 'good night' →", translate("good night"))
print("Translate 'thank you' →", translate("thank you"))
print("Translate 'i love you' →", translate("i love you"))

print("\n📜 Translation History")
print("-" * 40)
for i, (eng, fr) in enumerate(translation_history, 1):
    print(f"{i}. {eng}  →  {fr}")
print("-" * 40)

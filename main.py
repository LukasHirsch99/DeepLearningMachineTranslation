# 1. Start with token indices (from vocabulary)
tokens = [1, 42, 357, 2]  # [<sos>, "hello", "world", <eos>]

# 2. Look up embeddings
embed_layer = Embeddings(vocab_size=5000, d_model=512)
embeddings = embed_layer(tokens)
# Shape: [4, 512] - each token is now a 512-dim vector

# 3. Scale by sqrt(d_model)
embeddings = embeddings * sqrt(512)  # Done automatically in forward()

# 4. Add positional encoding (next step)
from positional_encoding import PositionalEncoding
pos_encoder = PositionalEncoding(d_model=512)
embeddings = pos_encoder(embeddings)

# 5. Now ready for transformer layers!
output = transformer(embeddings)

import nltk
# Define your desired weights (example: higher weight for bi-grams)
weights = (0.25, 0.25, 0, 0)  # Weights for uni-gram, bi-gram, tri-gram, and 4-gram

# Reference and predicted texts (same as before)
predictions = ["I", "am", "very", "grateful", "for", "chocolate", "."]
reference = ["I", "am", "eating", "chocolate", "."]

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']

#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], predictions)
print(BLEUscore)

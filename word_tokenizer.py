from nltk.tokenize import sent_tokenize,word_tokenize, WordPunctTokenizer

# Define input text
input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out."

# Divide the input text into sentence tokens
# Sentence tokenizer
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Divide the input text into word tokens:
# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# Divide the input text into word tokens using word punct tokenizer:
# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))


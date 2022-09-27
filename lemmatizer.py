from nltk.stem import WordNetLemmatizer

# define some input function
input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize','possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# Create lemmatizer object
lemmatizer = WordNetLemmatizer()

# Create a list of lemmatizer names for table display and format the text accordingly:
# Create a list of lemmatizer names for display
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names),'\n', '='*75)

# Iterate through the words and lemmatize the words using Noun and Verb lemmatizers:
# Lemmatize each word and display the output
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),
    lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))
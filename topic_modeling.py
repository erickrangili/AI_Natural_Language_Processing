from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

# Define a function to load the input data. The input file contains 10 line-separated sentences:
def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])
    return data

# Processor function for tokenizing, removing stop
# words, and stemming
def process(input_text):
    # Create a regular expression tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # We then need to stem the tokenized text:
    stemmer = SnowballStemmer('english')

    # We need to remove the stop words from the input text because they don't add information.
    # Let's get the list of stop-words:
    stop_words = stopwords.words('english')

    # Tokenize the input string
    tokens = tokenizer.tokenize(input_text.lower())

    # Remove the stop words
    tokens = [x for x in tokens if not x in stop_words]

    # Perform stemming on the tokenized words
    tokens_stemmed = [stemmer.stem(x) for x in tokens]
    return tokens_stemmed

# Define the main function and load the input data from the file data.txt provided to you:
if __name__=='__main__':
    # Load input data
    data = load_data('data.txt')

    # Create a list for sentence tokens
    tokens = [process(x) for x in data]

    # Create a dictionary based on the sentence tokens
    dict_tokens = corpora.Dictionary(tokens)

    # Create a document-term matrix
    doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]

    # Define the number of topics for the LDA model
    num_topics = 2

    # Generate the LDA model
    ldamodel = models.ldamodel.LdaModel(doc_term_mat,num_topics=num_topics, id2word=dict_tokens, passes=25)
    
    num_words = 5
    print('\nTop ' + str(num_words) + ' contributing words to each topic:')
    for item in ldamodel.print_topics(num_topics=num_topics,num_words=num_words):
        print('\nTopic', item[0])
        # Print the contributing words along with their relativecontributions
        list_of_strings = item[1].split(' + ')
        for text in list_of_strings:
            weight = text.split('*')[0]
            word = text.split('*')[1]
            print(word, '==>', str(round(float(weight) * 100, 2)) + '%')

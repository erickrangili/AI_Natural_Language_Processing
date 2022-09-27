from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Define some input words:
input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize','possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# Create objects for Porter, Lancaster, and Snowball stemmers:
# Create various stemmer objects
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')


# Create a list of names for table display and format the output text accordingly:
#Create a list of stemmer names for display
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names),'\n', '='*68)

# Iterate through the words and stem them using the three stemmers:
# Stem each word and display the output
for word in input_words:
    output = [word, porter.stem(word),lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output))
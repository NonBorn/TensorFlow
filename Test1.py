import nltk
from pprint import pprint
from nltk import sent_tokenize, pos_tag
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.stem import StemmerI, RegexpStemmer, LancasterStemmer, ISRIStemmer, PorterStemmer, SnowballStemmer, RSLPStemmer
from nltk.stem import  WordNetLemmatizer
from nltk.tag import StanfordNERTagger
import os


#nltk.download('all')
#testing github

text = 'The Athens University of Economics and Business (AUEB) was originally founded in\
1920 under the name of Athens School of Commercial Studies.\
It was renamed in 1926 as the Athens School of Economics and Business, \
a name that was retained until 1989 when it assumed its present name, \
the Athens University of Economics and Business.It is the third oldest \
university in Greece and the oldest in the fields of economics and business. \
Up to 1955 the school offered only one degree in the general area of economics and \
commerce. In 1955 it started two separate programs leading to two separate degrees: \
one in economics and the other in business administration. In 1984 the school was \
divided into three departments, namely the Department of Economics, the Department of \
Business Administration and the Department of Statistics and Informatics.In 1989, the \
university expanded to six departments. From 1999 onwards, the university developed \
even further and nowadays it includes eight academic departments, offering eight \
undergraduate degrees, 28 master\'s degrees and an equivalent number of doctoral programs.'


'''
Sentence tokenization
'''
sentences = sent_tokenize(text)
pprint(sentences)

'''
word tokenization
'''
tokens = word_tokenize(text)
pprint(tokens)

'''
Counting words
'''
count = Counter(tokens)
pprint(count.most_common(10))

'''
h'
count = nltk.FreqDist(tokens)
'''

'''
Removing stopwords
'''
#nltk.download(u'stopwords')
filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)
pprint(count.most_common(10))

'''
Creating ngrams
'''
bigrams = [ gram for gram in ngrams(tokens, 2) ]
trigrams = [ gram for gram in ngrams(tokens, 3) ]
pprint(trigrams)


'''
Tokenization with a regex
'''

tokenizer = RegexpTokenizer(r'\w+')
tokens2 = tokenizer.tokenize(text)
pprint(tokens2)


'''
POS tagging
'''
#nltk.download('averaged_perceptron_tagger')
pos_tags = pos_tag(tokens)
pprint(pos_tags)

# nltk.help.upenn_tagset()
# nltk.help.upenn_tagset('CC')
# nltk.batch_pos_tag([['this', 'is', 'batch', 'tag', 'test'], ['nltk', 'is', 'text', 'analysis', 'tool']])

'''
Stemming
'''
#stemmer = WordNetLemmatizer()
#stemmer = LancasterStemmer()
#stemmer = SnowballStemmer('english')
stemmer = PorterStemmer()
stems = [  stemmer.stem(token) for token in tokens ]
pprint(stems)


''''
Lemmatization
'''
#nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('are'))
print(lemmatizer.lemmatize('is'))
print(lemmatizer.lemmatize('is', pos='n'))
print(lemmatizer.lemmatize('is', pos='v'))
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet
from google_wordmap import spell_check_dict
import re
import os
fdir = os.path.split(os.path.realpath(__file__))[0]
root = os.path.split(fdir)[0] # the path of the root directory

# Activate debug mode or not. If True, the pickles of features will not be overwritten.
DEBUG = False

# Path of the data folder
DATA_PATH = os.path.join(root, 'data')
# How many samples would you like to get from train data. Set to None if you want to load all the \
# data
NUM_TRAIN = None
# How many samples would you like to get from test data. Set to None if you want to load all the \
# data
NUM_TEST = None
# How many product descriptions would you like to get from product description data. Set to None if\
# you want to load all the data
NUM_DESC = None 
# How many product attributes would you like to get from product attribute data. Set to None if \
# you want to load all the data
NUM_ATTR = None

# Abbreviate the columns of input dataframes. Do NOT change the values of this dict.
# Only the keys of this dict are configurable (according to your data file)
RENAME_DICT = {"search_term":"q", "product_title":"t", "product_description":"d", "brand":"b"}

# Stemmer used in the program. Choose 'porter' or 'snowball'.
STEM_TYPE = 'porter'

# Path of Word2Vec pretrained model
W2V_PATH = os.path.join(root, 'tools-w2v', 'GoogleNews-vectors-negative300.bin')
W2V_PATH = os.path.join(root, 'tools-w2v', 'hd')

# Self-designed clean function. To be used to clean the raw text in search term, product \
# descriptions and attributes. It would be the first cleaning.
# Note: Do NOT change the name of this function.
def clean_func(s): 
    s = str(s)
    #s = spell_check_dict.get(s, s) 
    if s in spell_check_dict.keys():
        s = spell_check_dict[s]
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) # split words with a.A
    s = s.lower()
    s = s.replace("  "," ")
    s = s.replace(",","") # could be number / segment later
    s = s.replace("$"," ")
    s = s.replace("?"," ")
    s = s.replace("-"," ")
    s = s.replace("//","/")
    s = s.replace("..",".")
    s = s.replace(" / "," ")
    s = s.replace(" \\ "," ")
    s = s.replace("."," . ")
    s = re.sub(r"(^\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x "," xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*"," xbi ")
    s = s.replace(" by "," xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    s = s.replace("°"," degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v "," volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    s = s.replace("  "," ")
    s = s.replace(" . "," ")
    #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
    str_number = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6', \
                    'seven':'7','eight':'8','nine':'0'}
    s = (" ").join([str_number[z] if z in str_number else z for z in s.split(" ")])

    # w2v lemmatize
    #toker = TreebankWordTokenizer()
    #lemmer = wordnet.WordNetLemmatizer()       
    #tokens = toker.tokenize(s)
    #s = " ".join([lemmer.lemmatize(z) for z in tokens])
    return s


    
    
    
    
    
    
    
    
    
    

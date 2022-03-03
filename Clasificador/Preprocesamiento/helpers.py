# Helpers
  
def remove_punctuation(string):
    clean_sentence = " ".join(re.split('\W+', string))
    
    return clean_sentence.strip()



def tokenize(sentence):
    return re.split(r'[ \W\n\t]+', sentence)



def remove_stop_words(token):
    stop_words = stopwords.words('spanish')
    token_filtered = [w for w in token if not w in stop_words]

    return token_filtered

    #return ' '.join(token_filtered)


def convert_string_to_list(string):
  return string.split(",")



def add_custom_lemmas(text):

    d = pd.read_json('drive/MyDrive/MSDS/Tesis/files/custom_lemmas.json')
    #d = pd.read_json('custom_lemmas.json')
    d = dict(d.values)
    result = []
    token = text.split(' ')
    for w in token:
        lemma = w
        try:
            lemma = d[w]
        except:
            lemma = w
        
        result.append(lemma)

    return ' '.join(result) 
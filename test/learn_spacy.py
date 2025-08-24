import spacy

class tokenize:
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence): 
        doc = self.nlp(sentence)
        return [token.text for token in doc if token.text != " "]

en_tokenizer = tokenize("en_core_web_sm")
print(en_tokenizer.tokenizer("i love apple very much"))
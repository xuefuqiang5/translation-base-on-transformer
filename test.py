import spacy
import sys

model = "en_core_web_sm"
nlp = spacy.load(model)
text = "i love you"
doc = nlp(text)
print(nlp.pipe_names)
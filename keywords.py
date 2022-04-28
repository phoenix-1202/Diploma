import sys
from googletrans import Translator
from keybert import KeyBERT
import re

translator = Translator()
result = translator.translate(sys.argv[1])
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
keywords = kw_model.extract_keywords(result.text)
best_1 = keywords[0]
best_2 = keywords[1]
# print(best_1, best_2)
# print(len(best_1))

words = re.split('[\\. ,\\?!;:]', sys.argv[1])
pos = 0
for word in words:
    if len(word) == 0:
        continue
    en_word = translator.translate(word).text
    # print(word, en_word)
    if en_word == best_1[0] or en_word == best_2[0]:
        print(pos)
    pos += 1

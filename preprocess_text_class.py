import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball


class PreProcessRawText:
    processed_text = None
    def __init__(self, raw_text):
        global processed_text
        self.processed_text = raw_text

    def remove_punctuation(self):
        self.processed_text = re.sub('[^a-zA-ZæøåÆØÅ]+', ' ', self.processed_text)
        self.processed_text = self.processed_text.replace("  ", " ")


    def remove_stopwords(self):
        tokens = word_tokenize(self.processed_text)
        filtered_words = [word for word in tokens if word not in set(stopwords.words('norwegian'))]

        self.processed_text = ' '.join(filtered_words)

    def stem_text(self):
        tokens = word_tokenize(self.processed_text)
        norStem = snowball.NorwegianStemmer()

        stemmed_words = list()
        for word in tokens:
            stemmed_words.append(norStem.stem(word))

        self.processed_text = ' '.join(stemmed_words)

    def processed_text_lower(self):
        self.processed_text.lower()

# if __name__ == '__main__':
#     raw_text = "Medietilsynet varslet sanksjoner mot Radio Metro etter slukkingen av FM-nettet. I en pressemelding skiver de at Radio Metro har sendt på FM-nettet i Oslo-området etter at de skulle stoppet.Når Radio Metro nå fortsetter med sine sendinger i Oslo uten konsesjon, er det ulovlig kringkasting, skriver direktør Mari Velsand i pressemeldingen. NTB skriver at Radio Metro avventer behandlingen i ESA, fordi slukkingen av FM-nettet er klaget inn dit."
#     test = PreProcessRawText(raw_text)
#     test.remove_punctuation()
#     test.remove_stopwords()
#     test.stem_text()
#     print(test.processed_text)

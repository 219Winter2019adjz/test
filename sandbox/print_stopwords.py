from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS


with open('stopwords.txt', "w+") as file:
    for word in stop_words:
        file.write(word + ", ")

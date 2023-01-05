from imports import *

reviews_train = pd.read_csv('en_training_dataset.csv').astype(str)

absa_model = Sequential()
absa_model.add(Dense(512, input_shape=(6000,), activation='relu'))
absa_model.add((Dense(256, activation='relu')))
absa_model.add((Dense(128, activation='relu')))
absa_model.add(Dense(8, activation='softmax'))
# compile model
absa_model.compile(loss='categorical_crossentropy',
                optimizer='Adam', metrics=['accuracy'])

vocab_size = 6000  # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews_train.review)
reviews_tokenized = pd.DataFrame(
tokenizer.texts_to_matrix(reviews_train.review))

label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(
reviews_train.aspect_category)
encoded_y = to_categorical(integer_category)
absa_model.fit(reviews_tokenized, encoded_y, epochs=100, verbose=1)

# model architecture
sentiment_model = Sequential()
sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
sentiment_model.add((Dense(256, activation='relu')))
sentiment_model.add((Dense(128, activation='relu')))
sentiment_model.add(Dense(3, activation='softmax'))
# compile model
sentiment_model.compile(loss='categorical_crossentropy',
                    optimizer='Adam', metrics=['accuracy'])

# create a word embedding of reviews data
vocab_size = 6000  # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews_train.review)
reviews_tokenized = pd.DataFrame(
tokenizer.texts_to_matrix(reviews_train.review))

# encode the label variable
label_encoder_2 = LabelEncoder()
integer_sentiment = label_encoder_2.fit_transform(reviews_train.sentiment)
encoded_y = to_categorical(integer_sentiment)
sentiment_model.fit(reviews_tokenized, encoded_y, epochs=100, verbose=1)

test_reviews = [
    "Flexibility, stability, dynamism, good benefits.",
    "Many unprepared managers make it discourage sometimes from day to day, but depart from each area. In recent years there has been a reduction in the number of staff, and with this there was work overload, many overtime, most of which end up becoming hours bank.",
    "Failure to recognize the work of some employees is discouraging. Important company sectors like HR and Personal Department are disorganized and leave a lot to be desired.",
    "The freedom to choose the technology that we will use if the staff agrees that it is advantageous.",
    "Bad salary. No possibility of renegotiation."
    ]
# Aspect preprocessing
test_reviews = [review.lower() for review in test_reviews]
test_aspect_terms = []
for review in nlp.pipe(test_reviews):
    chunks = [(chunk.root.text)
                for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    test_aspect_terms.append(' '.join(chunks))
test_aspect_terms = pd.DataFrame(
    tokenizer.texts_to_matrix(test_aspect_terms))

# Sentiment preprocessing
test_sentiment_terms = []
for review in nlp.pipe(test_reviews):
    if review.is_parsed:
        test_sentiment_terms.append(' '.join([token.lemma_ for token in review if (
            not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
    else:
        test_sentiment_terms.append('')
test_sentiment_terms = pd.DataFrame(
    tokenizer.texts_to_matrix(test_sentiment_terms))

# Models output
test_aspect_categories = label_encoder.inverse_transform(
    absa_model.predict_classes(test_aspect_terms))
test_sentiment = label_encoder_2.inverse_transform(
    sentiment_model.predict_classes(test_sentiment_terms))
for i in range(5):
    print("Review " + str(i+1) + " is expressing a  " +
            test_sentiment[i] + " opinion about " + test_aspect_categories[i])

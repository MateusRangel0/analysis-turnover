from imports import *
import generate_dataset as generate_dataset
import matplotlib.pyplot as plt

generate_dataset.create_columns()
reviews_train = pd.read_csv('../data/en_training_dataset.csv').astype(str)

absa_model = Sequential()
absa_model.add(Dense(512, input_shape=(6000,), activation='relu'))
absa_model.add((Dense(256, activation='relu')))
absa_model.add((Dense(128, activation='relu')))
absa_model.add(Dense(9, activation='softmax'))
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

test_csv = pd.read_csv('../data/new_en_experiment_dataset.csv').astype(str)

test_reviews = [review.review.lower() for review in test_csv.itertuples()]
former_emp = [review.former_emp for review in test_csv.itertuples()]

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
positives = 0
negatives = 0
isFormerForNegatives = {"isFormer": 0, "notFormer": 0}
aspectsDic = {
  'pay': [0, 0], 
  'conditions': [0, 0], 
  'nature_of_work': [0, 0], 
  'promotion': [0, 0], 
  'supervision': [0, 0], 
  'rewards': [0, 0], 
  'coworkers': [0, 0], 
  'communication': [0, 0],
  'benefits': [0, 0]
}
test_aspect_categories = label_encoder.inverse_transform(
    absa_model.predict_classes(test_aspect_terms))
test_sentiment = label_encoder_2.inverse_transform(
    sentiment_model.predict_classes(test_sentiment_terms))
for i in range(len(test_reviews)-1):
    print("Review " + str(i+1) + " is expressing a " + test_sentiment[i] + " opinion about " + test_aspect_categories[i])
    if (test_sentiment[i] == 'positive'):
        positives+=1
        generate_dataset.generate_visual_dataset(3, test_aspect_categories[i], former_emp[i])
        generate_dataset.generate_quantitative_dataset(3, test_aspect_categories[i], former_emp[i])
    if (test_sentiment[i] == 'negative'):
        if former_emp[i] == "1":
          isFormerForNegatives["isFormer"] += 1
          aspectsDic[test_aspect_categories[i]][0] += 1
        elif former_emp[i] == "0":
          isFormerForNegatives["notFormer"] += 1
          aspectsDic[test_aspect_categories[i]][1] += 1
        negatives+=1
        generate_dataset.generate_visual_dataset(1, test_aspect_categories[i], former_emp[i])
        generate_dataset.generate_quantitative_dataset(1, test_aspect_categories[i], former_emp[i])

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

lightBlue = '#42A5F5'
blue = '#0d6683'
explode = (0, 0.1)
# Build pie charts
plt.figure(0)
width = 0.35
sizes = [negatives, positives]
colors = [lightBlue, blue]
fig, ax = plt.subplots(figsize =(10, 7))
labelsFormer = "Negativas", "Positivas"
rect = plt.bar(labelsFormer, sizes, color=colors)
autolabel(rect)
print(sizes)
plt.savefig("PositiveAndNegativeProportion.png")
plt.figure(1)
fig, ax = plt.subplots(figsize =(10, 7))
labelsFormer = "Funcionário Atual", "Ex-Funcionário"
sizesFormer = [isFormerForNegatives["notFormer"], isFormerForNegatives["isFormer"]]
rect = plt.bar(labelsFormer, sizesFormer, color=colors)
autolabel(rect)
# plt.pie(sizesFormer, labels = labelsFormer, explode=explode, autopct='%1.1f%%', colors=[lightBlue, blue])
plt.savefig("FormerProportionOnNegativeReviews.png")

# Build bar Chart
plt.figure(2)
labels = ['Pagamento', 'Condições', 'Natureza de Trabalho', 'Promoção', 'Supervisão', 'Recompensas',
 'Colegas', 'Comunicação','Benefícios']
valuesDic = list(aspectsDic.values())
formerValues = [element[0] for element in valuesDic]
notFormerValues = [element[1] for element in valuesDic]
df = pd.DataFrame(dict(graph=labels, n=formerValues, m=notFormerValues)) 
ind = np.arange(len(df))
width = 0.4
fig, ax = plt.subplots(figsize =(15, 9))
rects1 = ax.barh(ind, df.n, width, label='Ex-Funcionário', color=lightBlue)
rects2 = ax.barh(ind + width, df.m, width, label='Funcionário', color=blue)
ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend()
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 5)  
ax.bar_label(rects1, padding=6)
ax.bar_label(rects2, padding=6)
ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.2)
plt.xlim(0, 50)
ax.set_title('Avaliações negativas agrupadas por tipo')
ax.set_xlabel('Quantidade')
plt.savefig("AspectsFormerProportion.png")
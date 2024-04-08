from transformers import BertTokenizer, BertForSequenceClassification,BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import pipeline

# Assuming df_clean is your DataFrame with 'REVIEW', 'sentiment', and 'target' columns
texts = df_clean['REVIEW'].tolist()
actual_sentiments = df_clean['sentiment'].tolist()
actual_targets = df_clean['target'].tolist()

# BERT Base Model
model_name_b='bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name_b)
config = BertConfig.from_pretrained(model_name_b)
model = BertForSequenceClassification.from_pretrained(model_name_b)

def sentiment_labels(text):
    encoded_input = tokenizer(text, padding=True,truncation=True, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]

sentiment_labels(texts_sample)


#Zero Shot Sentiment Analysis
model_name = "sentiment-analysis"
model_revision = "1.0"

# Create a zero-shot classification pipeline
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
batch_size = 8

# Perform zero-shot classification in batches
num_texts = 5332
predicted_labels = []

for i in range(0, num_texts, batch_size):
    batch_texts = texts[i:i+batch_size]
    predictions = classifier(batch_texts, ["positive", "negative"], multi_label=False)
    predicted_labels.extend([prediction['labels'][0] for prediction in predictions])


accuracy = accuracy_score(actual_sentiments, predicted_labels)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(actual_targets, [1 if label == 'positive' else 0 for label in predicted_labels])
print("Confusion Matrix:")
print(conf_matrix)






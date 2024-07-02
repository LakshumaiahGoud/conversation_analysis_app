import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

st.title("Conversation Analysis App")

# Load dataset using Hugging Face datasets library
ds = load_dataset("LDJnr/Puffin")
df = pd.DataFrame(ds['train'])

# Extract human conversations
conversations = []
for i, row in df.iterrows():
    convo_list = row['conversations']
    if isinstance(convo_list, list):
        for turn in convo_list:
            if turn['from'] == 'human':
                conversations.append({
                    'conversation_no': i,
                    'value': turn['value']
                })

df_conversations = pd.DataFrame(conversations).head(300)

# Initialize tokenizer for truncation
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Truncate long text sequences
def truncate_text(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    truncated_text = tokenizer.convert_tokens_to_string(tokens)
    return truncated_text

# Apply truncation to 'value' column
df_conversations['value'] = df_conversations['value'].apply(lambda x: truncate_text(x, max_length=512))

# Topic modeling with LDA
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_conversations['value'])
lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda.fit(X)

# Assign topics to conversations
topic_assignments = lda.transform(X).argmax(axis=1)
df_conversations['topic'] = pd.Series(topic_assignments, index=df_conversations.index)

# Get the top words for each topic
def get_topic_names(lda_model, vectorizer, n_top_words=10):
    topic_names = {}
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_names[topic_idx] = " ".join(top_words)
    return topic_names

topic_names = get_topic_names(lda, vectorizer)
df_conversations['topic'] = df_conversations['topic'].apply(lambda x: topic_names.get(x, 'Misc'))

# Display topic counts
st.subheader("Table 1: Topic Counts")
topic_counts = df_conversations['topic'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
st.table(topic_counts)

# Sentiment analysis with new model
def analyze_sentiment(text):
    try:
        truncated_text = truncate_text(text, max_length=512)
        result = sentiment_pipeline(truncated_text)
        if result and len(result) > 0:
            return result[0]['label']
        else:
            return 'neutral'  # Default sentiment if the result is empty
    except IndexError:
        return 'neutral'  # Default sentiment for errors
    except Exception:
        return 'neutral'  # Default sentiment for errors

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Add progress tracking with tqdm
st.write("Performing sentiment analysis...")
for i, row in tqdm(df_conversations.iterrows(), total=len(df_conversations), desc="Processing sentiments"):
    sentiment_label = analyze_sentiment(row['value'])
    df_conversations.at[i, 'sentiment'] = sentiment_label

# Display sentiment counts
st.subheader("Table 2: Sentiment Counts")
sentiment_counts = df_conversations['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
st.table(sentiment_counts)

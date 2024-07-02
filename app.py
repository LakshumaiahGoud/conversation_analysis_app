import streamlit as st
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline
from datasets import load_dataset

st.title("Conversation Analysis App")

# Load dataset using Hugging Face datasets library
ds = load_dataset("LDJnr/Puffin")
df = pd.DataFrame(ds['train'])

conversations = []
for i, convo in enumerate(df.itertuples()):
    for turn in convo.turns:
        if turn['from'] == 'human':
            conversations.append({
                'conversation_no': i,
                'value': turn['value']
            })

df = pd.DataFrame(conversations)

# Topic modeling
model = BERTopic()
topics, _ = model.fit_transform(df['value'])
df['topic'] = topics
df['topic'] = df['topic'].apply(lambda x: 'Misc' if x == -1 else x)

# Sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")
df['sentiment'] = df['value'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# Display counts
st.header("Counts")
st.subheader("Topic Counts")
topic_counts = df['topic'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
st.table(topic_counts)

st.subheader("Sentiment Counts")
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
st.table(sentiment_counts)

# Display sessions
st.header("Sessions")
session_page = st.sidebar.number_input("Page number", min_value=1, value=1, step=1)
sessions_per_page = 50
start = (session_page - 1) * sessions_per_page
end = start + sessions_per_page

st.table(df[['conversation_no', 'topic', 'sentiment']].iloc[start:end])

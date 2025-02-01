from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
    multimodal_faithness,
    multimodal_relevance,
    summarization_score
)

from datasets import Dataset
from langchain_openai import AzureChatOpenAI
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="XXXX")

embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

dataset_dict = {
    "question":["What is the capital of France"],
    "answer":["Paris is China"],
    "contexts":[["Paris is the capital and most populous city of France"]],
    "reference":["Paris"],
    "reference_contexts":[["Paris"]]
}

dataset = Dataset.from_dict(dataset_dict)

metrics_all = [answer_relevancy, answer_correctness, answer_similarity, context_entity_recall, context_precision, context_recall, faithfulness, multimodal_faithness, multimodal_relevance, summarization_score]

results = evaluate(dataset, metrics=metrics_all, llm=llm, embeddings=embedding_model)
print(results)
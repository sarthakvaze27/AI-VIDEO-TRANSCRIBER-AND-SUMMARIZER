#Imports Hugging Face's pipeline utility for easy model usage.
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


#- Defines a function to summarize text using BART
def summarize_text(
        text:str,
        model_name:str = "facebook/bart-large-cnn",
        max_length: int=150,
        min_length: int=30
) -> str:
    
     try:
         #- Loads the text-generation pipeline with the specified model
         summarizer = pipeline("text-generation", model=model_name)
         #Runs the summarization with length constraints.
         summary = summarizer(
              f"Summarize this text: {text}",
              max_length = max_length,
              min_length = min_length,
              do_sample = False,
              truncation=True
         )
         #- Returns the summary text from the first result.
         return summary[0]['generated_text'].replace("Summarize this text: ", "")
     except Exception as e:
         # Fallback to text-classification for sentiment analysis if needed
         try:
             summarizer = pipeline("text-classification", model=model_name)
             result = summarizer(text)
             return f"Classification result: {result[0]['label']} - {result[0]['score']:.2f}"
         except Exception as e2:
             raise Exception(f"Both text-generation and text-classification pipelines failed. Original error: {e}, Fallback error: {e2}")

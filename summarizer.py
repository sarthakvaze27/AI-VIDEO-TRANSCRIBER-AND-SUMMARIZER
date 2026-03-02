#Imports Hugging Face’s pipeline utility for easy model usage.
from transformers import pipeline


#- Defines a function to summarize text using BART
def summarize_text(
        text:str,
        model_name:str = "facebook/bart-large-cnn",
        max_length: int=150,
        min_length: int=30
) -> str:
    
     #- Loads the summarization pipeline with the specified model
     summarizer = pipeline("summarization",model=model_name)
     #Runs the summarization with length constraints.
     #- do_sample=False ensures deterministic output.
     summary = summarizer(
          text,
          max_length = max_length,
          min_length = min_length,
          do_sample = False
     )
     #- Returns the summary text from the first result.
     return summary[0]['summary_text']
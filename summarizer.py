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
         #- Loads the summarization pipeline with the specified model
         summarizer = pipeline("summarization", model=model_name)
         #Runs the summarization with length constraints.
         #- do_sample=False ensures deterministic output.
         summary = summarizer(
              text,
              max_length = max_length,
              min_length = min_length,
              do_sample = False,
              truncation=True
         )
         #- Returns the summary text from the first result.
         return summary[0]['summary_text']
     except Exception as e:
         # Fallback to a different approach if pipeline fails
         try:
             summarizer = pipeline("text2text-generation", model=model_name)
             summary = summarizer(
                 f"summarize: {text}",
                 max_length=max_length,
                 min_length=min_length,
                 do_sample=False,
                 truncation=True
             )
             return summary[0]['generated_text']
         except Exception as e2:
             raise Exception(f"Both summarization and text2text-generation pipelines failed. Original error: {e}, Fallback error: {e2}")

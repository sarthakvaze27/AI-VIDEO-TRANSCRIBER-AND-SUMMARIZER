#- Splits long text into overlapping chunks.
def chunk_text(text: str, chunk_size: int=2000,overlap: int = 200) -> list:
    #- Initializes variables for chunking.
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start+ chunk_size,text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

        if start < 0:
            start = 0

    return chunks       
    
 #- Summarizes long text in stages.
def chunked_summarize(text: str, summarize_func,max_chunk_size: int = 2000) -> str:
    #1.To split the text into chunks
    text_chunks = chunk_text(text,chunk_size=max_chunk_size,overlap=200)
    
    #2.To summarize each chunk separately(indivisually)
    partial_summaries = [summarize_func(chunk) for chunk in text_chunks]

    #3.To combine partial summaries
    combined_summary_input = " ".join(partial_summaries)

    #4.To run a final summarization step on the combined partial summaries
    final_summary = summarize_func(combined_summary_input)
    return final_summary
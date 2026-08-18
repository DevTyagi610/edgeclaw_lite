from typing import List

def build_rag_prompt(query : str, chunks : List[dict]) -> str :
    # Empty chunks case 
    sys_prompt = """You are a helpful assistant. Answer the question using ONLY the context below. If the answer is not in the context, say you don't know. """
    if len(chunks) == 0:
        return sys_prompt + "\n\nContext:\n There is no relevant context found"  + "\nQuestion: " + query + "\n\nAnswer: " 

    context = ""
    for i in range(0, len(chunks)):
        source = str(chunks[i]["source"])
        chunk_id = str(chunks[i]["chunk_id"])
        context = context + "[" + source + "]"+  "["+  chunk_id + "] "+ chunks[i]["text"] + "\n"

    prompt = sys_prompt + "\n\nContext:\n" + context + "\nQuestion: " + query + "\n\nAnswer: " 
    return prompt
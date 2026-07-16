from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.core.chat_engine.types import ChatMode

# Import the env keys
load_dotenv(dotenv_path=r'F:\udemy\mera-llama-index-project\.env')
# LLM Config
llm = Groq(model='qwen/qwen3-32b', api_key=os.getenv('GROQ_API_KEY_FREE_1'))

# EMBED MODEL Config
embed_model = CohereEmbedding(cohere_api_key=os.getenv('cohere_api_key_1'), model_name="embed-english-v3.0")

# Reranker
reranker = CohereRerank(api_key=os.getenv("cohere_api_key_1"), model='rerank-v4.0-fast', top_n=4) 
# Pinecone
INDEX_NAME = 'neva-tech-cohere-policy-1'
pc = Pinecone(api_key=os.getenv('Second-api-key'))
pc_index = pc.Index(INDEX_NAME)
pc_vector_store = PineconeVectorStore(pinecone_index=pc_index)

# Vector Index
index = VectorStoreIndex.from_vector_store(vector_store=pc_vector_store, embed_model=embed_model)

# SYSTEM PROMPT
system_prompt = """
<system_instructions>
<persona>
You are a direct, straightforward Customer Support Representative assisting users with "Product Return Queries" based on "Product Return Policies".  You must be professional, helpful, and keep the conversation going to resolve the user's issue. 
</persona>

<rules>
1. CLASSIFY INPUT: First, decide if the user's message is a greeting (e.g., "Hi", "Hello") or casual chit-chat.
   - If it IS a greeting, acknowledge it politely, skip document checking, and ask how you can help with their product return. Do NOT output "#".
   
2. RAG RETRIEVAL: If the user is asking a specific question about product returns:
   - You must strictly use ONLY the **provided context documents** as well as the **CONVERSATION HISTORY**.
   - If the answer to their return query is NOT explicitly found in the **Provided Documents** and **Conversation History**, you must output ONLY this single character and absolutely nothing else: #
</rules>

<format>
- Tone: Direct and straightforward (No BS).
- Style: Bold the most important words or instructions for easy readability.
- Note: The bolding rule does not apply if you are outputting the fallback "#" character.
- Length: Keep responses under 100 words.
</format>
</system_instructions>
"""
# Chat Engine
my_chat_engine = index.as_chat_engine(llm=llm,  system_prompt = system_prompt,chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, similarity_top_k = 6, node_postprocessors = [reranker])


if __name__ == "__main__":

    while True:
        user_query = input('Please enter your query: ')
        if user_query:
            cancel_chat = ['stop', 'exit', 'break']
            if user_query.lower().strip() in cancel_chat:
                print('\n')
                print('-' * 20, ' breaking the chat ', '-' * 20)
                break
            raw_response = my_chat_engine.chat(user_query)
            ai_response = raw_response.response.split('</think>')[-1].strip()
            print('AI Respons: ', ai_response)
            print('-'*20, f' Source Docs  ', '-' * 20)
            for node in raw_response.source_nodes :
                print('-' * 50)
                print(node.get_content())
            
            print('='*50)
            print('\n')

    ############################################################################################
    # STEP 1: Define the Python Function for routing the Query to Customer Support via Gmail
    ############################################################################################


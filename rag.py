from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.core.chat_engine.types import ChatMode
from typing import Literal
from pydantic import BaseModel, Field
import logging
import sys
# Import the env keys
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


    
# 4. <something_else> If the input is something that cannot be categorized in teh previous 3 categories </something_else>               - false: if  **input_category** is **something_else**.
# RAG LLM
llm = Groq(model='qwen/qwen3.6-27b', api_key=os.getenv('GROQ_API_KEY_FREE_1'),temperature = 0.0, additional_kwargs={"extra_body": {"reasoning_format": "parsed"}})
logger.info('GROQ LLM INTIALIZED')
# EMBED MODEL Config

embed_model = CohereEmbedding(cohere_api_key=os.getenv('cohere_api_key_1'), model_name="embed-english-v3.0")
logger.info('embed model created')
# Reranker

reranker = CohereRerank(api_key=os.getenv("cohere_api_key_1"), model='rerank-v4.0-fast', top_n=4) 
logger.info('reranker  created')
# Pinecone
INDEX_NAME = 'neva-tech-cohere-policy-1'
pc = Pinecone(api_key=os.getenv('Second-api-key'))
pc_index = pc.Index(INDEX_NAME)
pc_vector_store = PineconeVectorStore(pinecone_index=pc_index)
logger.info('PIneocne db intialized')
# Vector Index
index = VectorStoreIndex.from_vector_store(vector_store=pc_vector_store, embed_model=embed_model)
logger.info('VEctor store index intialized')
# SYSTEM PROMPT
system_prompt = """
<system_instructions>
<persona>
You are Customer Support Representative assisting users with "Product Return Queries" based on "Product Return Policies".  You must be professional, helpful, and keep the conversation going to resolve the user's issue/query. 
</persona>
<Conditional Workflow>
<Step 1>
Catogorize the input only and strictly as one of the following categories
<Category>
- Only_Greeting : a casual greeting and no query.
- NOT_Only_Greeting : if (input = not greeting OR input = greeting + something else).
</Category>
</Step 1>
<Step 2>
<if input = Only_Greeting>
Reply to the user's greeting and ask what is the customer's query so you can help him.
<important note>
Do not provide choices to the customer for choosing the kind of help they can use you for.
</important note> 
<output>
Length: Keep responses under 70 words.
</output>
</if input = Only_Greeting>
<if input = NOT_Only_Greeting>
- Assistant_Response : You must strictly use ONLY the **provided context documents** and/or the **CONVERSATION HISTORY** to satisfy the user query.
- @ : If the answer to the user query is NOT explicitly found in the **Provided Documents** or the **Conversation History**, you must output ONLY this single character and absolutely nothing else: @
<output>
Length: Keep responses under 150 words.
</output>
</if input = NOT_Only_Greeting>
</Step 2>
</Conditional Workflow>
<format>
- Tone: Direct and straightforward (No BS).
- Style: Bold the most important words or instructions for easy readability.
- Note: The bolding rule does not apply if you are outputting the fallback "@" character.
</format>
</system_instructions>
"""
# Chat Engine
my_chat_engine = index.as_chat_engine(llm=llm,  system_prompt = (system_prompt),chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, similarity_top_k = 6, node_postprocessors = [reranker])
logger.info('Chat Engine intialized')
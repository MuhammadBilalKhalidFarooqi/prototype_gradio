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
# Import the env keys
load_dotenv(dotenv_path=r'F:\udemy\mera-llama-index-project\.env')

# Raw LLM Config
raw_llm = Groq(model='qwen/qwen3.6-27b', api_key=os.getenv('GROQ_API_KEY_FREE_1'), strict = True)

# Pydantic Validation
class Structred_LLM_Output(BaseModel):

    thought_process: str = Field(
        description="""Explain your step-by-step reasoning based on the "system prompt" rules before making a decision."""
    )
    input_category : Literal['Greeting', 'Customer_Query', 'Greeting_and_Query'] = Field(
        description="Category of the user's input"

                        
                        
                    
    )
    is_answer_found : bool = Field(
        description="""
                - true : if input_category is Greeting OR the answer to the user_input is found in the **Provided Context Documents** and/or **Conversation History**.
                - false : otherwise
                """        
        
    )
    
          
       
  
    Response: str | Literal['answer_not_available'] = Field(
        description="""
        The final response to the user
        """
    )
# 4. <something_else> If the input is something that cannot be categorized in teh previous 3 categories </something_else>               - false: if  **input_category** is **something_else**.
# RAG LLM
llm = raw_llm.as_structured_llm(output_cls=Structred_LLM_Output)

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
You are Customer Support Representative assisting users with "Product Return Queries" based on "Product Return Policies".
You must be professional, helpful, and keep the conversation going to resolve the user's issue/query. 
</persona>
<instructions>
1. Categorize the user's input
2. If "input_category" is  "Greeting" then no need to search the **Provided Context Documents**/**Conversation History** BUT if "input_category" is "Query/Greeting_and_Query" then answer strictly based on **Provided Context Documents** and/or **Conversation History**. Do not guess the answer (non-negotiable)
3. answer_not_available : if "is_answer_found" is "false" | response : The final response to the user, if "is_answer_found" is "true".
</instructions>
</system_instructions>
"""
# Chat Engine
def support_bot(user_input):
    raw_llm
    my_chat_engine = index.as_chat_engine(llm=llm,  system_prompt = system_prompt,chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, similarity_top_k = 6, node_postprocessors = [reranker])



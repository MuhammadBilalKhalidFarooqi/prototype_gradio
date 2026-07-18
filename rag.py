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
raw_llm = Groq(model='qwen/qwen3-32b', api_key=os.getenv('GROQ_API_KEY_FREE_1'), strict = True)

# Pydantic Validation
class Structred_LLM_Output(BaseModel):
    input_category : Literal['Greeting', 'Query', 'Greeting_and_Query'] = Field(
        description=""" Categorize the user's input:

                        1. Greeting : if input is Normal Greeting/chit chat. 
                        2. Query: if input is a Genuine Query/Support.  
                        3. Greeting_and_Query : if input contains a Normal Greeting as well as a User Query.
                        
                    """
    )
    escalatation_status:  bool = Field(
        description=""" Choose exactly one status based on the result of "input_category":
                        - true : if **input_category** is  **Greeting**.
                        - true : if **input_category** is  **Query** and the answer to the user's input is avaible in the **Provided Context Documents** and/or **Conversation History**.
                        - true : if **input_category** is  **Greeting_and_Query** and the answer to the user's input is available in the **Provided Context Documents** and/or **Conversation History**.
                        - false : if **input_category** is  **Greeting_and_Query** and the answer to the user's input is **NOT** available in the **Provided Context Documents** and/or **Conversation History**.
                        - false : if **input_category** is  **Query** and the answer to the user's input is **NOT** available in the **Provided Context Documents** and/or **Conversation History**.
                      """   
       )                   
          
       
  
    Response: str | Literal['answer_not_available'] = Field(
        description="""
        
        - answer_not_available : if "escalation_status" is "false".
        - string(a response to entertain the user's input) : If "escalation_status" is "true"
        
        
        <format for response>
        - Keep the text under 150 words.
        - Bold the most important terms or instructions.
        - Be direct , professional and straightforward (No BS).
        </format for response>
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
If the 'validation rules' allow to do so then you must be professional, helpful, and keep the conversation going to resolve the user's issue/query . 
</persona>
</system_instructions>
"""
# Chat Engine
my_chat_engine = index.as_chat_engine(llm=llm,  system_prompt = system_prompt,chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, similarity_top_k = 6, node_postprocessors = [reranker])



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
    input_category : Literal['Greeting', 'Query', 'Greeting and Query', 'something else'] = Field(
        description=""" 1. <Greeting> if input == Greeting/chit chat.</Greeting> 
                        2. <Query> if input == User Query </Query> 
                        3. <Greeting and Query> if input == Greeting as well as a User Query.</Greeting and Query>
                        4. <something else> if the specific class for user's input is not the previous three</something else>
                    """
    )
    escalate_to_human: bool | None | Literal['bad_input']= Field(
        description="""1. <null> if Field **input_category** == Greeting </null> 
                        2.<true> if the answer to the user's query is not availble in the **Provided Context Documents** or **Conversation History**</true> 
                        3.<false> if the answer to the user's query is avaible in the **Provided Context Documents** or **Conversation History**</false>
                        4.<bad_input> if Field **input_category** == something else</bad_input>
        """
    )
    Response : str | None= Field(
        description= """
            <null> if Field **escalate_to_human** == true</null>
            <str> if Field **escalate_to_human** == false. Entertain/Satisfy the user's query<str>
        """
    )

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
You are Customer Support Representative assisting users with "Product Return Queries" based on "Product Return Policies".  You must be professional, helpful, and keep the conversation going to resolve the user's issue/query. 
</persona>
<format>
- Tone: Direct and straightforward (No BS).
- Style: Bold the most important words or instructions for easy readability.
- Length: Keep responses under 150 words.
</format>
</system_instructions>
"""
# Chat Engine
my_chat_engine = index.as_chat_engine(llm=llm,  system_prompt = system_prompt,chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, similarity_top_k = 6, node_postprocessors = [reranker])



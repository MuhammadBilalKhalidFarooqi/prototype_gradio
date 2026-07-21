import gradio as gr
from rag import my_chat_engine
from llama_index.core.llms import ChatMessage, MessageRole
import os
def my_chatbot(message, history):
    chat_history = []
    for msg in history:
        if msg['role'] == 'user':
            user_msg = msg['content'][0]['text']
            chat_history.append(ChatMessage(role =MessageRole.USER , content= user_msg)) 

        elif msg['role'] == 'assistant':
            ai_msg = msg['content'][0]['text']
            chat_history.append(ChatMessage(role = MessageRole.ASSISTANT, content = ai_msg))
        else:
            raise ValueError('there is an error in the chat history loop')

    response = my_chat_engine.chat(message=message, chat_history=chat_history).response #.split('</think>')[-1].strip()

    
    return response


demo = gr.ChatInterface(fn=my_chatbot, save_history=True, textbox=gr.Textbox(placeholder='enter your query hurry'))

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
    

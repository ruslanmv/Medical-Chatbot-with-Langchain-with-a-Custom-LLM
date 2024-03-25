## Building a Medical Chatbot with Langchain and custom LLM via API
Hello everyone, today we are going to build a simple  Medical Chatbot by using a Simple Custom LLM.
This blog post explores how to construct a medical chatbot using Langchain, a library for building conversational AI pipelines, and Milvus, a vector similarity search engine and a remote custom remote LLM via API. The combination allows the chatbot to retrieve relevant information from a medical conversation dataset and leverage a large language model (LLM) service to generate informative responses to user queries.

###  1. Step 1: Dataset Retrieving 
This part of the program retrieves a dataset for medical conversation. It uses the load_dataset function from the datasets library to load a dataset named "ai-medical-chatbot" from the Hugging Face Hub.
It then cleans the question and answer columns and prepares the data for further use.
```python
from datasets import load_dataset

dataset = load_dataset("ruslanmv/ai-medical-chatbot")
train_data = dataset["train"]

# For this demo, let's choose the first 1000 dialogues
df = pd.DataFrame(train_data[:1000])
df = df[["Description", "Doctor"]].rename(columns={"Description": "question", "Doctor": "answer"})

# Clean the question and answer columns
df['question'] = df['question'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
df['answer'] = df['answer'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
```

The code snippet above retrieves a medical conversation dataset named "ai-medical-chatbot" from the Hugging Face Hub. It then selects the first 1000 dialogues and cleans the question and answer columns to prepare the data for further processing.

### 2. Step 2: Milvus Connection 
This part establishes a connection to a Milvus vector store. Milvus is a vector similarity search engine that can be used to efficiently search for similar vectors.
It loads a collection named "qa_medical" from the Milvus server.

```python
from dotenv import load_dotenv
import os
from pymilvus import connections

load_dotenv()
COLLECTION_NAME='qa_medical'

host_milvus = os.environ.get("REMOTE_SERVER", '127.0.0.1')
connections.connect(host=host_milvus, port='19530')

collection = Collection(COLLECTION_NAME)      
collection.load(replica_number=1)
```

This code snippet establishes a connection to a Milvus vector store running on the specified host and port. It then loads a collection named "qa_medical" from the Milvus server. This collection likely stores pre-encoded medical conversations that can be efficiently searched for similar questions.

### 3. Step 3: Custom LLM (Large Language Model)

This section defines a custom LLM component. While the provided code doesn't reveal the specific LLM service used, it demonstrates how to wrap an external LLM service with custom functions for formatting prompts and handling responses.

### 4. Step 4: Langchain Definitions 
This part defines the core functionalities of the medical chatbot using Langchain, a library for building conversational AI pipelines.
It defines a custom retriever class CustomRetrieverLang that retrieves relevant answers from the Milvus vector store for a given question.
It defines functions full_prompt and custom_llm that prepare the prompts for the LLM and call the custom LLM service to generate responses.
Finally, it creates a Langchain instance rag_chain that wraps the custom LLM functionalities.
```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.milvus import Milvus

class CustomRetrieverLang(BaseRetriever):
     def get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Perform the encoding and retrieval for a specific question
        # (implementation omitted for brevity)
        return [Document(page_content=answer_string)]

retriever = CustomRetrieverLang()

def full_prompt(question, history=""):
    # Get the retrieved context from Milvus (implementation omitted for brevity)
    system_message = "..."  # Add system message template here
    formatted_prompt = format_prompt_zephyr(question, history, system_message=system_message)
    return formatted_prompt

def custom_llm(question, history="", temperature=0.8, max_tokens=256, top_p=0.95, stop=None):
    formatted_prompt = full_prompt(question, history)
    # Call LLM service with formatted prompt (implementation omitted for brevity)
    return output
```

This section defines the core functionalities of the medical chatbot using Langchain. Here's a breakdown of the code:

* **CustomRetrieverLang:** This class retrieves relevant answers from the Milvus vector store for a given question. (The specific implementation for encoding and retrieval is omitted for brevity).
* **full_prompt:** This function prepares the prompt for the LLM service by incorporating retrieved context from Milvus and system messages.
* **custom_llm:** This function utilizes the `full_prompt` function to format the prompt and then interacts with the external LLM service to generate responses.

Finally, a Langchain instance named `rag_chain` is created to encapsulate these functionalities.

## 5. Define your chat function (Code Snippet)

This part defines the chat function that interacts with the Langchain model to generate responses for user queries.
It takes a message and a history of conversation as input and returns the updated history and the generated response.

```python
from langchain.llms import BaseLLM
from langchain_core.language_models.llms import LLMResult

class MyCustomLLM(BaseLLM):
  def _generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 256, top_p: float = 0.95, stop: list[str] = None, **kwargs) -> LLMResult:
    response_text = custom_llm(question=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop)
    response = LLMResult(generations=[[{'text': response_text}]])
    return response

  def _llm_type(self) -> str:
    return "Custom LLM"

# Create a Langchain with your custom LLM
rag_chain = MyCustomLLM()
```

This section defines the `chat` function, which serves as the interface between the user and the Langchain model. Here's what the code does:

* **MyCustomLLM:** This class inherits from Langchain's `BaseLLM` class. It overrides the `_generate` method to call the `custom_llm` function (defined earlier) and convert the response into the LLMResult format expected by Langchain.
* **rag_chain:** An instance of `MyCustomLLM` is created, essentially creating a Langchain model that utilizes the custom LLM functionality.

## 6. Create a Gradio Interface

The final step involves creating a user-friendly interface for interacting with the medical chatbot. This is achieved using Gradio, a library for building web interfaces for machine learning models.

Here's the code snippet for the Gradio interface:

```python
import gradio as gr

def chat(message, history):
  response = rag_chain.invoke(message)
  return response

# Create a Gradio interface
with gr.Blocks() as interface:
  # Display a welcome message (implementation omitted for brevity)
  with gr.Row():
    with gr.Column():
      text_prompt = gr.Textbox(label="Input Prompt", placeholder="Example: What are the symptoms of COVID-19?", lines=2)
      generate_button = gr.Button("Ask Me", variant="primary")
  with gr.Row():
    answer_output = gr.Textbox(type="text", label="Answer")
  generate_button.click(chat, inputs=[text_prompt], outputs=answer_output)

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=7860)
```

The code defines two functions:

* **chat:** This function takes a message and conversation history as input, interacts with the Langchain model to generate a response, and updates the history.


Gradio components are then used to create a user interface with:

* A text box for users to enter their medical questions.
* A button to trigger the generation of a response.
* Another text box to display the response from the chatbot.

By clicking the button, users initiate the `chat` function, which interacts with the Langchain model and displays the generated response. Finally, the Gradio interface is launched, making the medical chatbot accessible through a web browser.

## Conclusion

This blog post has explored the development of a medical chatbot using Langchain and Milvus. The approach leverages pre-existing medical conversation data stored in Milvus for context retrieval and integrates an external LLM service for response generation. The resulting chatbot offers a user-friendly interface built with Gradio, allowing users to interact with the model and receive informative responses to their medical queries.

It's important to remember that this is a simplified example, and real-world medical chatbots would require additional considerations such as:

* **Medical Disclaimers:**  Clearly stating that the chatbot is not a substitute for professional medical advice.
* **Data Quality and Biases:** Ensuring the training data used for the LLM is high-quality and free from biases.
* **Safety and Accuracy:** Implementing safeguards to prevent the chatbot from providing inaccurate or misleading information.

By carefully addressing these aspects, this approach has the potential to contribute to the development of informative and helpful medical chatbots. 
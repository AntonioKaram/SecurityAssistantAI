import os
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate

ZEPHYR_ID = "HuggingFaceH4/zephyr-7b-beta"

def get_model(repo_id=ZEPHYR_ID, **kwargs):
    hf_token = kwargs.get("HUGGINGFACEHUB_API_TOKEN", None)

    if not hf_token:
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    os.environ["HF_TOKEN"] = hf_token
    
    llm = HuggingFaceHub(
        repo_id=repo_id,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
            "huggingfacehub_api_token": hf_token,
        })
    return llm
    # chat_model = ChatHuggingFace(llm=llm)
    # return chat_model

def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template("Hello world")

    chain = prompt | model
    return chain

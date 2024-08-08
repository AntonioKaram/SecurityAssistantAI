import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from models import get_model
from loader import load_data
from retriever import create_retriever

st.set_page_config(page_title="OSSA Assistant")
st.title("OSSA AI Assistant")

def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )

    return response

def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "ai", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "ai":
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)['answer']

                answer = response[response.find('Helpful Answer:'):]

                st.markdown(answer) #or .content
        message = {"role": "ai", "content": answer}
        st.session_state.messages.append(message)

def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    hf_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    if not hf_api_token:
        hf_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                        info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")
            
    if not hf_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        st.stop()

    model = get_model(HUGGINGFACEHUB_API_TOKEN=hf_api_token)
    memory = ConversationBufferMemory(llm=model, memory_key="chat_history", return_messages=True, output_key='answer')

    docs = load_data()
    retriever = create_retriever(docs)

    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, memory=memory, return_source_documents=True)

    st.subheader("Ask me questions about the OSSA policy")
    show_ui(chain, "What would you like to know?")

run()
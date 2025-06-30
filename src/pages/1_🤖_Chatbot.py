import uuid
from typing import Generator

import streamlit as st

from basic_rag.rag import graph


def response_generator(graph, prompt) -> Generator:
    config = {"configurable": {"thread_id": st.session_state['thread_id']}}
    for token, metadata in graph.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        stream_mode="messages"
    ):
        if not token.type == "tool":
            yield token


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = uuid.uuid4()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(graph, prompt=prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

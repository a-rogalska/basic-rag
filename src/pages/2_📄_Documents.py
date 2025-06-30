import streamlit as st

from basic_rag.doc_stores import chroma_store
from basic_rag.extractors import docling_extractor

uploaded_file = st.file_uploader("Upload new document", type=("jpg", "jpeg", "pdf", "png"))

if "documents" not in st.session_state:
    st.session_state["documents"] = []

if uploaded_file:
    with st.spinner("Uploading documents..."):
        st.write(f"Processing file: {uploaded_file.name}")  # Display the name of the uploaded file
        # Extract text from the uploaded document
        extracted_text = docling_extractor.extract(uploaded_file.name, uploaded_file.read())

        # Add the extracted text to the document store
        doc_id = uploaded_file.name  # Use the file name as the document ID
        chroma_store.add_document(id=doc_id, document_content=extracted_text)
        st.session_state["documents"].append(
            {
                "filename": uploaded_file.name,
                "content": extracted_text,
            }
        )

        st.success(f"Document '{uploaded_file.name}' has been successfully added to the store.")

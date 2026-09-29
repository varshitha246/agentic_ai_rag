import streamlit as st

from src.graph import rag_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f7f8fc;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .main-header h1 {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-header p {
        color: #6b7280;
        font-size: 16px;
    }

    /* Chat messages */
    .user-message {
        background: #e8f0fe;
        padding: 14px 18px;
        border-radius: 14px;
        margin: 10px 0;
    }

    .assistant-message {
        background: white;
        padding: 16px 18px;
        border-radius: 14px;
        margin: 10px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Source cards */
    .source-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 12px;
        margin: 6px 0;
        font-size: 13px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Agentic AI RAG Assistant</h1>
        <p>Ask questions about the Agentic AI reference material</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 About")

    st.write(
        """
        This chatbot uses:

        - 🧠 LangGraph
        - 🔎 Pinecone
        - 📚 Retrieval-Augmented Generation
        - 🤗 Sentence Transformers
        - ⚡ Groq LLM
        """
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        **1. Question**

        Your question is received.

        **2. Retrieval**

        Relevant chunks are retrieved from Pinecone.

        **3. Generation**

        The LLM generates an answer using the retrieved reference material.

        **4. Grounding**

        If the answer is not supported by the reference material,
        the chatbot says so instead of guessing.
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                <b>You</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="assistant-message">
                <b>🤖 Assistant</b><br><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display sources if available
        if message.get("sources"):

            with st.expander("📖 View retrieved sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Page:</b> {source.get("page", "N/A")}
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            <b>Score:</b> {source.get("score", 0):.4f}
                            <br><br>
                            {source.get("text", "")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Ask me anything about Agentic AI from the provided reference material."
    )

    st.markdown("### Try asking")

    examples = [
        "What is Agentic AI?",
        "What are the building blocks of Agentic AI?",
        "What is multi-agent collaboration?",
        "How does Agentic AI differ from traditional AI?",
    ]

    for question in examples:

        if st.button(question, use_container_width=True):

            st.session_state.pending_question = question

            st.rerun()


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Agentic AI..."
)


# Handle example question
if "pending_question" in st.session_state:

    question = st.session_state.pop("pending_question")


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching the reference material..."):

            try:

                result = rag_graph.invoke(
                    {
                        "question": question
                    }
                )

                # ------------------------------------------------
                # Get final answer
                # ------------------------------------------------

                answer = result.get("answer", "")

                if not answer:

                    answer = (
                        "I could not generate an answer from the "
                        "provided reference material."
                    )

                st.markdown(answer)

                # ------------------------------------------------
                # Get retrieved documents
                # ------------------------------------------------

                retrieved_docs = result.get(
                    "documents",
                    []
                )

                sources = []

                for doc in retrieved_docs:

                    # Pinecone result
                    if hasattr(doc, "metadata"):

                        metadata = doc.metadata or {}

                        sources.append(
                            {
                                "page": metadata.get(
                                    "page",
                                    "N/A"
                                ),
                                "score": metadata.get(
                                    "score",
                                    0
                                ),
                                "text": doc.page_content,
                            }
                        )

                    # Dictionary result
                    elif isinstance(doc, dict):

                        sources.append(
                            {
                                "page": doc.get(
                                    "page",
                                    "N/A"
                                ),
                                "score": doc.get(
                                    "score",
                                    0
                                ),
                                "text": doc.get(
                                    "text",
                                    ""
                                ),
                            }
                        )

                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

                # ------------------------------------------------
                # Show sources
                # ------------------------------------------------

                if sources:

                    with st.expander(
                        "📖 View retrieved sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"""
                                <div class="source-card">

                                <b>Page:</b>
                                {source["page"]}

                                &nbsp;&nbsp;|&nbsp;&nbsp;

                                <b>Score:</b>
                                {source["score"]:.4f}

                                <br><br>

                                {source["text"]}

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            except Exception as e:

                st.error(
                    "Something went wrong while processing your question."
                )

                st.exception(e)
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from groq import AuthenticationError, RateLimitError, APIConnectionError
import os

#load .env 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 
    
def validate_groq_api():
    """Validate GROQ_API_KEY exists; skip gracefully if missing"""
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. QA chains will not run.")
        return False
    try:
        test_llm = ChatGroq(model_name="Llama3-8b-8192", max_tokens=1)
        test_llm.invoke("test")
        return True
    except (AuthenticationError, APIConnectionError, RateLimitError):
        print("Warning: GROQ_API_KEY invalid or connection failed. QA chains may fail.")
        return False

def format_docs(docs: list[Document]) -> str:
    """Format documents for the context prompt."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

def format_chat_history(chat_history: list) -> str:
    """Format chat history into a string."""
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])


def create_qa_chain(retriever):
    """Create an enhanced QA + Analysis chain"""
    has_key = validate_groq_api()

    llm = ChatGroq(
        model_name="Llama3-70b-8192",
        temperature=0.5,
        max_tokens=2048,
        max_retries=3,
        request_timeout=15
    ) if has_key else None

    # Context prompt
    CONTEXT_PROMPT = PromptTemplate(
        template="""
        [INST] Answer strictly based only on the context below. 
        Clearly indicate which source each piece comes from (PDF name or YouTube).
        If multiple sources are available, cross-reference them.
        If unsure, say "This isn't clear from the sources.
        Context: {context}
        Question: {question} [/INST]
        """,
        input_variables=["context", "question"]
    )

    # Analysis prompt
    ANALYSIS_PROMPT = PromptTemplate(
        template="""
        [INST] Based on the following question and the context answer provided, 
        generate your own analysis that goes beyond the given sources. 
        This could include:
        - Connecting concepts between sources
        - Drawing broader implications
        - Identifying potential gaps or limitations
        - Providing educated guesses where appropriate
        - Offering practical insights or recommendations
        Question: {question}
        Context Answer: {context_answer}
        Provide your analysis below: [/INST]
        """,
        input_variables=["question", "context_answer"]
    )

    # Identity prompt to classify if the user is asking for the bot's name
    IDENTITY_PROMPT = PromptTemplate.from_template(
        """
        [INST] You are a classification model. Your task is to determine if the user's question is asking for the AI's name, identity, or who it is.
        Respond with only "YES" or "NO". Do not add any other text.

        User Question: {question}
        Is the user asking for your name or identity? (YES/NO) [/INST]
        """
    )

    # Condense question prompt
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """
        [INST] Given the following conversation and a follow up question, rephrase the 
        follow up question to be a standalone question, in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question: [/INST]
        """
    )

    if llm:
        # Use a smaller, faster model for classification tasks
        classifier_llm = ChatGroq(
            model_name="Llama3-8b-8192",
            temperature=0.0,
            max_tokens=5, # "YES" or "NO"
        ) if has_key else None

        # This chain condenses the chat history and the new question into a standalone question
        standalone_question_chain = (
            {
                "question": lambda x: x["query"],
                "chat_history": lambda x: format_chat_history(x["chat_history"])
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
        )

        # This chain classifies if the question is about identity
        identity_classifier_chain = (
            {"question": lambda x: x["query"]}
            | IDENTITY_PROMPT
            | classifier_llm
            | StrOutputParser()
        )

        # This chain determines which question to use for retrieval (original or condensed)
        # It's a branch that checks if chat_history exists.
        question_router_chain = RunnableBranch(
            (lambda x: x.get("chat_history"), standalone_question_chain),
            (lambda x: x["query"])
        )

        # Step 1: Define the inputs for the RAG chain, retrieving documents in parallel.
        rag_inputs_chain = RunnableParallel(
            {
                "context": question_router_chain | retriever | format_docs,
                "question": question_router_chain,
                "source_documents": question_router_chain | retriever,
            }
        )

        # Step 2: Generate the context-based answer. `RunnablePassthrough.assign` passes
        # the original inputs through and adds the new `context_answer` key.
        chain_with_context_answer = RunnablePassthrough.assign(
            context_answer=(CONTEXT_PROMPT | llm | StrOutputParser())
        )

        # Step 3: Generate the analysis answer. This step receives the output from the
        # previous step (including `context_answer`) and adds the `analysis_answer`.
        chain_with_analysis = RunnablePassthrough.assign(
            analysis_answer=(ANALYSIS_PROMPT | llm | StrOutputParser())
        )

        # Step 4: Combine all steps into the final, sequential chain.
        rag_chain = rag_inputs_chain | chain_with_context_answer | chain_with_analysis

        # This function provides the hardcoded response when the bot is asked its name
        def get_identity_response(inputs):
            return {
                "identity_response": True, # Flag for special handling in the UI
                "answer": "My name is Lumi. I'm an AI-powered study assistant designed to help you analyze and understand your study materials. You can upload PDFs and YouTube videos, and I'll help you find answers and insights from them.",
                "source_documents": [] # Keep for schema consistency
            }

        # The main chain that routes between the identity response and the RAG chain
        main_chain = RunnableBranch(
            (
                lambda x: "yes" in identity_classifier_chain.invoke(x).lower(),
                RunnableLambda(get_identity_response)
            ),
            rag_chain # Fallback to the original RAG chain
        )

        def enhanced_qa_invoker(inputs):
            """Wrapper to handle potential errors and match expected output format."""
            try:
                return main_chain.invoke(inputs)
            except Exception as e:
                print(f"Error during chain invocation: {e}")
                return {
                    "context_answer": "LLM failed or API key missing.",
                    "analysis_answer": "LLM failed or API key missing.",
                    "source_documents": []
                }

        return enhanced_qa_invoker
    else:
        # No API key: return dummy retriever
        def dummy_qa(inputs):
            return {
                "context_answer": "GROQ API key missing, cannot answer.",
                "analysis_answer": "GROQ API key missing, cannot answer.",
                "source_documents": []
            }
        return dummy_qa

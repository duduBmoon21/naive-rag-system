from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    if "GROQ_API_KEY" not in os.environ:
        print("Warning: GROQ_API_KEY not set. QA chains will not run.")
        return False
    try:
        test_llm = ChatGroq(model_name="Llama3-8b-8192", max_tokens=1)
        test_llm.invoke("test")
        return True
    except (AuthenticationError, APIConnectionError, RateLimitError):
        print("Warning: GROQ_API_KEY invalid or connection failed. QA chains may fail.")
        return False

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

    # Wrap RetrievalQA
    if llm:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": CONTEXT_PROMPT, "verbose": True},
            return_source_documents=True
        )
        analysis_chain = ANALYSIS_PROMPT | llm

        def enhanced_qa(inputs):
            try:
                qa_result = qa_chain.invoke({"query": inputs["query"]})
                analysis_result = analysis_chain.invoke({
                    "question": inputs["query"],
                    "context_answer": qa_result["result"]
                })
                return {
                    "context_answer": qa_result["result"],
                    "analysis_answer": analysis_result.content,
                    "source_documents": qa_result["source_documents"]
                }
            except Exception as e:
                return {
                    "context_answer": "LLM failed or API key missing",
                    "analysis_answer": "LLM failed or API key missing",
                    "source_documents": []
                }

        return enhanced_qa
    else:
        # No API key: return dummy retriever
        def dummy_qa(inputs):
            return {
                "context_answer": "GROQ API key missing, cannot answer.",
                "analysis_answer": "GROQ API key missing, cannot answer.",
                "source_documents": []
            }
        return dummy_qa

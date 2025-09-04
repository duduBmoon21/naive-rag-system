import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

parser = StrOutputParser()

# ----------------------------
# PROMPTS
# ----------------------------
IDENTITY_PROMPT = PromptTemplate.from_template("""
[INST] Is the user asking about your identity (your name, who you are, what you do)?
Answer YES or NO only.
User: {question} [/INST]
""")

GREETING_PROMPT = PromptTemplate.from_template("""
[INST] Is the user simply greeting (e.g., hi, hello, hey)?
Answer YES or NO only.
User: {question} [/INST]
""")

CONTEXT_PROMPT = PromptTemplate.from_template("""
[INST] Answer strictly from the context. 
Context:
{context}

Question:
{question} [/INST]
""")

ANALYSIS_PROMPT = PromptTemplate.from_template("""
[INST] Provide deeper analysis of context.
Context:
{context}

Question:
{question} [/INST]
""")

# ----------------------------
# BUILD CHAIN
# ----------------------------
def build_chain(retriever, llm=None, reranker=None):
    """
    Returns a RunnableLambda that handles identity, greeting, context, and analysis.
    Expects retriever to have get_relevant_documents().
    """
    # Load LLM
    if llm is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing. Please set in .env or env variables.")
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, max_tokens=2048)

    # Identity branch
    identity_chain = RunnableLambda(
        lambda inputs: "YES" in llm.invoke(
            IDENTITY_PROMPT.format_prompt(question=inputs["query"])
        ).content.upper()
    )

    # Greeting branch
    greeting_chain = RunnableLambda(
        lambda inputs: "YES" in llm.invoke(
            GREETING_PROMPT.format_prompt(question=inputs["query"])
        ).content.upper()
    )

    # Retrieve context
    def retrieve_context(inputs):
        docs = retriever.get_relevant_documents(inputs["query"])
        context_str = "\n\n".join([f"{d.page_content}" for d in docs])
        return {"context": context_str, "source_documents": docs, "question": inputs["query"]}

    # Context chain
    context_chain = RunnableLambda(
        lambda inputs: llm.invoke(
            CONTEXT_PROMPT.format_prompt(**retrieve_context(inputs))
        ).content
    )

    # Analysis chain
    analysis_chain = RunnableLambda(
        lambda inputs: llm.invoke(
            ANALYSIS_PROMPT.format_prompt(**retrieve_context(inputs))
        ).content
    )

    # Branching
    def classify_and_route(inputs):
        q = inputs["query"]

        if identity_chain.invoke({"query": q}):
            return {"identity_response": True, "answer": "I’m Lumi, your study buddy powered by RAG!"}

        if greeting_chain.invoke({"query": q}):
            return {"identity_response": True, "answer": "Hey there 👋 I’m Lumi! Ask me about your sources."}

        context_answer = context_chain.invoke({"query": q})
        analysis_answer = analysis_chain.invoke({"query": q})
        sources = retrieve_context({"query": q})["source_documents"]

        return {
            "identity_response": False,
            "context_answer": context_answer,
            "analysis_answer": analysis_answer,
            "source_documents": sources
        }

    return RunnableLambda(classify_and_route)

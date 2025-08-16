from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from groq import AuthenticationError, RateLimitError, APIConnectionError
import os

def validate_groq_api():
    """Validate GROQ_API_KEY exists and is valid"""
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY missing from environment variables")
    
    try:
        test_llm = ChatGroq(model_name="Llama3-8b-8192", max_tokens=1)
        test_llm.invoke("test")
    except AuthenticationError:
        raise ValueError("Invalid GROQ_API_KEY - check at console.groq.com")
    except APIConnectionError:
        raise ConnectionError("Failed to connect to Groq servers")
    except RateLimitError:
        raise RuntimeError("Rate limit exceeded - try again later")

def create_qa_chain(retriever):
    try:
        # 1. Validate API key first
        validate_groq_api()
        
        # 2. Initialize LLM 
        llm = ChatGroq(
            model_name="Llama3-70b-8192",  
            temperature=0.5, 
            max_tokens=2048,  
            max_retries=3,
            request_timeout=15
        )
        
        # 3. Context prompt 
        context_prompt_template = """
        [INST] Answer strictly based only on the context below. 
        Clearly indicate which source each piece comes from (PDF name or YouTube).
        If multiple sources are available, cross-reference them.
        If unsure, say "This isn't clear from the sources."

        Context: {context}

        Question: {question} [/INST]
        """
        
        CONTEXT_PROMPT = PromptTemplate(
            template=context_prompt_template,
            input_variables=["context", "question"]
        )
        
        # 4. Analysis prompt 
        analysis_prompt_template = """
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
        """
        
        ANALYSIS_PROMPT = PromptTemplate(
            template=analysis_prompt_template,
            input_variables=["question", "context_answer"]
        )
        
        # 5. Create the context answer chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": CONTEXT_PROMPT,
                "verbose": True  
            },
            return_source_documents=True
        )
        
        # 6. Create analysis chain
        analysis_chain = ANALYSIS_PROMPT | llm
        
        def enhanced_qa(inputs):
            # First get context-based answer
            qa_result = qa_chain.invoke({"query": inputs["query"]})
            
            # Then generate analysis
            analysis_result = analysis_chain.invoke({
                "question": inputs["query"],
                "context_answer": qa_result["result"]
            })
            
            return {
                "context_answer": qa_result["result"],
                "analysis_answer": analysis_result.content,
                "source_documents": qa_result["source_documents"]
            }
        
        return enhanced_qa
        
    except AuthenticationError as e:
        raise ValueError(f"Groq authentication failed: {str(e)}")
    except RateLimitError as e:
        raise RuntimeError(f"Groq rate limit exceeded: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize QA chain: {str(e)}")
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from groq import AuthenticationError, RateLimitError, APIConnectionError
import os

def validate_groq_api():
    """Validate GROQ_API_KEY exists and is valid"""
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY missing from environment variables")
    
    # Test with minimal request
    try:
        test_llm = ChatGroq(model_name="Llama3-8b-8192", max_tokens=1)
        test_llm.invoke("test")  # Verify API connectivity
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
        
        # 2. Initialize LLM with your preferred settings
        llm = ChatGroq(
            model_name="Llama3-8b-8192",
            temperature=0.3,
            max_tokens=1024,
            max_retries=3,  # Auto-retry on transient failures
            request_timeout=15  # Fail fast if unresponsive
        )
        
        # 3. Your custom prompt template (preserved)
        prompt_template = """
        [INST] Answer strictly based on the context below. 
        If unsure, respond "I'm really sorry this is beyond your context"

        Context: {context}

        Question: {question} [/INST]
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 4. Chain creation with error wrapping
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True  
            },
            return_source_documents=True
        )
        
    except AuthenticationError as e:
        raise ValueError(f"Groq authentication failed: {str(e)}")
    except RateLimitError as e:
        raise RuntimeError(f"Groq rate limit exceeded: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize QA chain: {str(e)}")
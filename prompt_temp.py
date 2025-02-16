from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from constants import *

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Let's think step by step.
Always say "thanks for asking!" at the end of the answer.
{context}

Question: {question}

Helpful Answer:"""

template1 = """ answer the question in the end and Generate a detailed process chain if necessary from the given context. 
    Provide the process chain in points and ensure each step is clearly and concisely described. 
    Use relevant technical terms and specify any important details. 

    **Instructions:**
    1. Read the context carefully.
    2. Identify all the key steps involved in the electric motor manufacturing process.
    3. Break down the process into sequential points.
    4. Include any necessary sub-steps or details that are crucial to understanding the process.
    5. Make sure the steps are ordered logically from start to finish.
    6. Keep the headings in bold.

    **Example Context:**
    The context is related to the manufacturing process of electric motors, starting from the raw material procurement to the final assembly and testing of the motor. It involves steps such as winding of the stator, rotor assembly, insulation testing, and final quality checks.

    **Example Process Chain:**

    1. **Raw Material Procurement**
    - Source high-quality copper wire for winding.
    - Procure steel for the stator and rotor laminations.

    2. **Stator Winding**
    - Cut and shape the steel laminations for the stator.
    - Wind the copper wire around the stator laminations to form the stator coils.

    3. **Rotor Assembly**
    - Cut and shape the steel laminations for the rotor.
    - Assemble the rotor core and attach the shaft.

    4. **Insulation and Impregnation**
    - Insulate the stator coils with varnish.
    - Perform vacuum pressure impregnation to ensure durability.

    5. **Assembly**
    - Insert the rotor into the stator.
    - Assemble additional components such as bearings and end shields.

    6. **Electrical Testing**
    - Conduct insulation resistance tests.
    - Perform high-voltage tests to ensure no short circuits.

    7. **Performance Testing**
    - Test the motor under load conditions to ensure proper operation.
    - Measure parameters like torque, speed, and efficiency.

    8. **Final Quality Checks**
    - Inspect the motor for any physical defects.
    - Perform a final run test to ensure all specifications are met.

    **Context:**
    {context}

    Question: {question}"""

# Create a custom prompt template for retrieval-augmented generation (RAG)
custom_rag_prompt = ChatPromptTemplate.from_template(template1)


# Function to generate an answer using the RAG chain
def prompt_answer(query, retriever):
    """
    Generate an answer for the given query using a retrieval-augmented generation chain.

    Parameters:
    - query (str): The user's query or question.
    - retriever: The retriever object to fetch relevant context.

    Returns:
    - str: The generated response from the RAG chain.
    """
    # Define the RAG chain sequence:
    # 1. Retrieve context using the retriever.
    # 2. Apply the custom RAG prompt template.
    # 3. Use the language model (LLM) for generating responses.
    # 4. Parse the output to extract the final answer.
    rag_chain = (
        {
            "context": retriever,  # Retrieve relevant context
            "question": RunnablePassthrough(),  # Pass the original query as-is
        }
        | custom_rag_prompt  # Apply custom RAG prompt
        | llm  # Use the language model for response
        | StrOutputParser()  # Parse the response as a string
    )

    # Invoke the RAG chain with the provided query and return the result
    return rag_chain.invoke(query)

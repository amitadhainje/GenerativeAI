import os
from constants import openai_api_key
from langchain_openai import ChatOpenAI
import streamlit as st # We can also use Flask
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize the StreamLit Framework
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want")

llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    PromptTemplate.from_template("Tell me about celebrity {name} in 5 lines")
    | llm
    | StrOutputParser()
    | PromptTemplate.from_template("when was the celebrity born:\n{text}")
    | llm
    | StrOutputParser()
    | PromptTemplate.from_template("Mention 5 major events which happened in that {text}")
    | llm   
    | StrOutputParser()
)

if input_text:
    result = chain.invoke(input_text)
    st.write(result)

# Trail Code is as follows -
# llm = OpenAI(temperature=0.8)
###
# temperature = This suggests that how much control the agent should have while
# giving you the response. 
###
# METHOD-1 - START - the following code is the first method to club two prompts and chains together. 
# first_prompt = PromptTemplate(
#     input_variables=['name'],
#     template="Tell me about celebrity {name} "
# )

# first_chain = first_prompt | llm | StrOutputParser()
# first_chain_with_key = first_chain | {"summary": RunnablePassthrough()}

# second_prompt = PromptTemplate(
#     input_variables=["summary"],
#     template="when was the celebrity born:\n{summary}"
# )

# second_chain = second_prompt | llm | StrOutputParser()
# final_chain = first_chain_with_key | second_chain
# METHOD - 1 - END

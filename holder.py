import dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

from flask import Flask, request, jsonify
import subprocess



dotenv.load_dotenv()
app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    
    data = request.get_json()
    print(data, type(data))
    chroma_path = "chroma_data/"

    reviews_vector_db = Chroma(
        persist_directory=chroma_path,
        embedding_function=OpenAIEmbeddings()
    )

    reviews_retriever = reviews_vector_db.as_retriever(k=10)



    chat_model= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    review_template_str = """Act as if You are an AI assistant tasked with analyzing email drafts
    to assess the need for a meeting.Provide recommendations based on the content of the email.
    As an AI consultant specializing in workplace efficiency,evaluate the email draft to determine
    if scheduling a meeting is the most effective course of action.You are an automated meeting scheduler
    designed to optimize productivity. Review the email and advise whether a meeting is warranted or 
    if there are alternative communication methods."


    {context}
    """

    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template_str,
        )
    )

    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )
    messages = [review_system_prompt, review_human_prompt]

    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    output_parser = StrOutputParser()

  
    question = "Analyze the following email with above context as a sample if meeting is required or not." + data['question']


    review_chain = {"context":reviews_retriever, "question": RunnablePassthrough()}| review_prompt_template | chat_model | output_parser
    suggestion = review_chain.invoke(question)
    output = {
        "message": suggestion,
    }
    print(suggestion)
    
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=5001)
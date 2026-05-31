from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RannableLamda

from dotenv import load_dotenv
load_dotenv()

# 1 load your vector db
BASE_DIR = Path(__file__).resolve().parent

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  

vectorstore = Chroma(
    collection_name="bajaj_policy",
    embedding_function=embedding_model,
    persist_directory=BASE_DIR / "bajajbot_chroma_db",
)


# you can define your retrival stretgies

retreival = vectorstore.as_retriever(
   search_type = 'mmr',
   search_kwargs={'k': 3,
                  'fetch_k': 3,
                  "lambda_mult": 0.7,
                  'threshold': 0.9
                    }
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Bajaj Finance helpdesk agent.
Answer the customer query using ONLY the policy context below.

STRICT RULES:
- Use ONLY the POLICY CONTEXT provided — never use general knowledge
- If answer is not in context, say exactly:
  "I don't have information on this. Please contact Rahul @ 9152091676."
- Never say "based on general practices" or "typically"
- Never make up numbers, rates, or policies
- Format amounts with Rs and commas (e.g., Rs 8,450)
- "minimum CIBIL score" means the score required for APPROVAL

POLICY CONTEXT:
{context}"""),

    # Chat history goes here — LLM sees previous turns for follow-up context
    # This is SEPARATE from retrieval — ChromaDB never sees this history
    MessagesPlaceholder(variable_name="chat_history"),

    # Only the current query — clean, no history mixed in
    ("human", "{query}"),
])





rag_chain = prompt | retreival | llm





# query = "who is PM of india?"

# docs = retreival.invoke(query)

# for doc in docs:
#     print("##"*20)
#     print(doc.page_content)
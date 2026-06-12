# need 3 session to finish entire projects in depth
hireflow/                 
├── config.py              
├── .env                  
├── requirements.txt      
│
├── api/
│   ├── main.py
│   └── routes/
│       ├── upload.py
│       ├── evaluate.py
│       └── actions.py
│
ingestion/ # Plan in session 1
        ├── loader.py        ← PDF bulk reader
        ├── chunker.py       ← Section-aware splitting
        ├── metadata.py      ← Email extract + LLM fallback
        ├── dedup.py         ← Pinecone email check + overwrite
        ├── embedder.py      ← OpenAI text-embedding-3-small
        ├── vector_store.py  ← Pinecone upsert
         pipeline.py    
        ├── jd_loader.py     ← JD ingestion
        ├── tasks.py         ← Celery tasks
        └── celery_app.py    ← Celery + Redis config
├── retrieval/
        ├── query_embedder.py   ← JD ko embed karo
        ├── retriever.py        ← Pinecone se candidates fetch karo
        ├── reranker.py         ← bge-reranker-base se top-k refine karo
        └── chain.py            ← sab ko ek pipeline mein connect karo
generation/
├── prompt.py      ← evaluation prompt template
├── llm.py         ← GPT-4o-mini setup
├── parser.py      ← structured output (Pydantic)
└── eval_chain.py  ← full chain assembly

api/
├── main.py        ← FastAPI app
└── routes/
    └── evaluate.py ← POST /evaluate → SSE stream
└── scripts/
evaluation/
├── ragas_dataset.py    ← evaluation dataset builder
└── ragas_runner.py     ← RAGAS scores compute karo

observability/
├── mlflow_setup.py     ← experiment + tracking setup
├── logger.py           ← metrics + artifacts log karo
└── tracer.py           ← @mlflow.trace on eval chain

pip install \
  "langchain-core==0.3.28" \
  "langchain==0.3.12" \
  "langchain-openai==0.2.14" \
  "langchain-community==0.3.12" \
  "llama-index-core" \
  "llama-index-readers-file" \
  "llama-index-embeddings-openai" \
  "pinecone" \
  "sentence-transformers" \
  "fastapi" \
  "uvicorn[standard]" \
  "python-multipart" \
  "celery" \
  "redis" \
  "openai" \
  "mlflow" \
  "prometheus-fastapi-instrumentator" \
  "pydantic-settings" \
  "python-dotenv" \
  "pymupdf" \
  "datasets"
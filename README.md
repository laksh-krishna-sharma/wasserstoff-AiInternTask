# QueryQuill

An interactive chatbot powered by AI that ingests, processes, and researches across large document sets (75+ docs), identifies common themes, and delivers precise, cited answers to user queries—all with a rich UI and Docker support.

## 🚀 Features

- **Multi-document ingestion** (75+ documents supported)
- **Thematic analysis** using NLP techniques
- **Interactive chatbot interface** with citation
- **Fast API backend** (FastAPI + Uvicorn)
- **Dockerized** for easy deployment

## Getting Started

### Prerequisites

- Python 3.11.9
- Docker

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/laksh-krishna-shrarma/wasserstoff-AiInternTask.git
   cd wasserstoff-AiInternTask
   ```

2. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running Locally

Start the development server:

```bash
uvicorn app.main:app --reload
# or via docker
docker compose up -d
```

Open your browser and go to [http://localhost:8090](http://localhost:8090)

## 📚 Usage

### Ingest documents
- Place .pdf/.txt files into `./data/`
- Run ingestion endpoint or script (e.g., `POST /ingest`)

### Explore thematically
- Run `GET /themes` to list extracted themes

### Chat
- Use `/chat` endpoint to ask questions—get detailed, cited answers referencing your documents

## 🧩 Architecture

```
┌───────────────┐       ┌──────────────┐       ┌──────────────┐
│   FastAPI     │ <─>   │   Chatbot    │ <─>   │   Vector DB  │
│   Endpoints   │       │   Logic +    │       │ + NLP Engine │
│  (Chat/Ingest)│       │   LLM calls  │       │              │
└───────────────┘       └──────────────┘       └──────────────┘
```

- **FastAPI** handles endpoint logic (`/ingest`, `/themes`, `/chat`)
- **Chatbot Core**: 
  - Loads documents, turns content into vectors (FAISS), extracts themes (LDA/transformers)
  - Routes user queries through: retrieval → theme detection → LLM response generation
  - Emits citations referencing original documents
- **Storage**: Documents + metadata + vector indexes persisted locally or in Docker volumes

## 🔧 Configuration

Edit `config.yaml` (or `.env`) to customize:

- `DATA_DIR`: where documents are stored
- NLP parameters: theme count, vector dimension
- LLM model & token settings
- Docker-compose ports

## 📈 Roadmap / To-Do

- 🔍 Expand ingestion formats (`.docx`, `.epub`)
- 🧠 Support multi-turn conversation with context memory
- 📊 Visualize themes via charting
- 🔐 Add authentication + permissions
- 🌐 Deploy to cloud (AWS/GCP)

## 🤝 Contributing

All contributions welcome! Just:

1. Fork the repo
2. Create a feature branch
3. Submit a PR with your changes

We'll review and merge!

## 👍 Contact

Have questions, issues, or feature ideas? Open an issue or email:
asklakshsharma@gmail.com

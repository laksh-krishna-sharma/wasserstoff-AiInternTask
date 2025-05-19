
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
    pythom -m venv venv
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

Open your browser and go to [http://localhost:8090]
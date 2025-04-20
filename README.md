# Listic Lite

AI-powered recipe ingredient extractor and potential shopping list optimizer.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd listic-lite
    ```

2.  **Install Poetry:**
    If you don't have Poetry installed, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Create Environment File:**
    Create a file named `.env` in the project root directory. Add your OpenAI API key to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `your_openai_api_key_here` with your actual key.

4.  **Install Dependencies:**
    Poetry will automatically create a virtual environment (in the project's `.venv` directory) and install the required packages.
    ```bash
    poetry install
    ```

5.  **Install Playwright Browsers:**
    Playwright requires browser binaries to function. Install them (including OS dependencies) using:
    ```bash
    poetry run playwright install --with-deps
    ```

## Running the Script

To run the main script which uses the AI agent to process recipes:

```bash
poetry run python main.py
```

This command executes the `main.py` script within the Poetry-managed virtual environment.

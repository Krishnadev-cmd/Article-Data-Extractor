
# News Research Tool 📈

This Streamlit application allows users to input URLs of news articles, process the content, and query the articles using a language model. The application leverages Google's Gemini model and FAISS vector stores to provide answers to user queries based on the content of the provided URLs.

## Features

- **URL Input**: Users can input up to three URLs of news articles.
- **Text Processing**: The application loads and processes the content from the provided URLs.
- **Text Splitting**: The content is split into manageable chunks for efficient processing.
- **Embeddings Creation**: The application uses HuggingFace's embeddings to convert the content into vector representations.
- **Vector Store**: FAISS (Facebook AI Similarity Search) is used to store the vectorized content for efficient retrieval.
- **Query Interface**: Users can input a question, and the app will retrieve relevant information from the processed articles, providing an answer along with the sources.

## Installation

To run this application, you need to have Python installed. Follow the steps below:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/news-research-tool.git
    cd news-research-tool
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - Create a `.env` file in the root directory.
    - Add your Google API key in the `.env` file:
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## How to Use

1. **Enter URLs**: In the sidebar, enter up to three URLs of the news articles you want to analyze.
2. **Process URLs**: Click the "Process URLs" button to load and process the content.
3. **Ask Questions**: Enter a query in the text input box and press Enter. The app will display the answer along with the sources.

## Dependencies

- `streamlit`: Used for building the web application.
- `pickle`: For saving and loading the FAISS vector store.
- `langchain_google_genai`: Integrates Google's Gemini model.
- `langchain_huggingface`: For generating embeddings using HuggingFace models.
- `FAISS`: For efficient similarity search and retrieval.
- `dotenv`: For loading environment variables.

## Future Improvements

- Allow users to input more than three URLs.
- Add error handling for invalid URLs or empty responses.
- Improve the UI for displaying answers and sources.




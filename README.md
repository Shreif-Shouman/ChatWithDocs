# 📚 Chat with Your Documents

Welcome to the **Chat with Your Documents** app! This Streamlit application allows you to upload multiple documents (PDF, DOCX, and TXT), process them, and chat with the content using advanced language models. It's a perfect tool for students, researchers, and anyone who needs to extract and interact with information from documents.

## 🛠 Features

- **Multiple File Formats**: Supports PDF, DOCX, and TXT files.
- **Advanced Language Models**: Utilizes cutting-edge language models for understanding and answering questions.
- **User-Friendly Interface**: Easy to use with a clean and intuitive Streamlit interface.
- **Interactive Chat**: Ask questions and get answers from your documents.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.7+
- `pip` (Python package installer)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/chat-with-documents.git
    cd chat-with-documents
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your Hugging Face API token:
        ```env
        HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
        ```

### Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Upload Documents**:
    - Use the sidebar to upload your PDF, DOCX, and TXT files.
    - Click on "Process" to extract and analyze the content.

3. **Chat with Your Documents**:
    - Ask questions in the chat input box.
    - Get responses based on the content of your uploaded documents.

## 📝 Example

Here's a quick example of how to use the app:

1. Upload your documents.
2. Ask a question like "What are the main points in the document?".
3. Receive detailed answers extracted from your documents.

## 📦 File Structure


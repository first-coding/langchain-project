## Intelligent Q&A System Based on LangChain, FAISS, and LLM

### Project Overview
This project implements an intelligent Q&A system based on LangChain, FAISS, and LLM. The system's features include:
- **File Search**: Supports file uploads, vectorizes the files, and performs semantic search using FAISS to find relevant content.
- **LLM-based Q&A**: The system utilizes LLM to generate answers based on the retrieved information and context.

### Installation

### Clone the repository
```bash
git clone git@github.com:first-coding/langchain-project.git
```
### Install required dependencies
```bash
pip install -r requirements.txt
```
PS: If the installation is slow, you can try using a mirror.

### Run the project
```bash
python main.py
```

### Notes
1.Due to limited computational resources, I am using a smaller model and running it on a CPU.

2.Remember to initialize the system when uploading files, otherwise errors may occur.

3.Users with inconsistent Python versions may encounter version-related errors.

4.If you want to run the system on a GPU, modify main.py, cls/langchain_application.py, and gpt_service.py, and install the corresponding CUDA version.

### Suggestions and Issues
I hope this project is helpful to everyone，certainly，This project is a reproduction and modification based on the LangChain code from GitHub. If you have any suggestions or issues, feel free to discuss them with me through the issues section. Thank you!"
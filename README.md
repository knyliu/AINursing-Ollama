# AI Nursing - Nursing Clinical Decision Education System

## Demonstration
A Quick Demo of AI Nursing - Clinical Decision-Making Education System: 2 Practical Cases

https://youtu.be/xioUZLmTWoE

## Project Description

This open-source software is designed to help healthcare institutions and related fields utilize their historical case reports by embedding the data into a vector database, enabling the system's functionality. The integration of AI capabilities makes the data more scalable and valuable, making it suitable for clinical decision education and training.

## Project Directory Structure

The following is the basic directory structure of this project based on the Flask framework. The homepage is `index.html`, the main system is in `main.html`, and `about.html` and `data.html` are reserved for future expansion.

```
AI_Nursing/
│
├── faiss_index/         # Location for storing the vector database
├── templates/
│   ├── index.html       # Homepage
│   ├── main.html        # Main system page
│   ├── about.html       # About page
│   └── data.html        # Data page
├── app.py               # Flask main program
├── requirements.txt     # Dependencies file
└── README.md            # Project description file
```

## Installation Guide

### 1. Clone the Project

```bash
git clone <repository-url>
cd AI_Nursing
```

### 2. Install Required Packages

The project uses Flask, OpenAI API, LangChain, and FAISS. Install the following dependencies:

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File

Prepare your OpenAI API Key and set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage Instructions

A small vector database has already been prepared in the `faiss_index` folder for user convenience. Due to copyright and intellectual property restrictions, we are unable to provide more data. However, you can still use this small database for testing. If you wish to replace it with your own data, simply use FAISS to convert the prepared data into a vector database and replace the original file.

In addition, this system relies on the OpenAI API as the underlying language model. Please ensure that you have set up the API Key.

### Start the Application:

```bash
python app.py
```

The browser will open the homepage at `http://127.0.0.1:5000/`.

## Contact Information

For any questions, please contact:

- Email: 41171123h@ntnu.edu.tw
- Email: pecu@ntnu.edu.tw
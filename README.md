# Vietnamese Sentiment Analysis Project

This project aims to perform sentiment analysis on Vietnamese text using the [TextCNN](https://arxiv.org/pdf/1408.5882) model. TextCNN is a convolutional neural network model specifically designed for sentence classification tasks, making it suitable for analyzing sentiments based on Vietnamese language texts.
### Dataset
https://github.com/congnghia0609/ntc-scv
## Features
- **Vietnamese Language Processing**: Tailored for processing and analyzing Vietnamese text data.
- **TextCNN Architecture**: Utilizes convolutional neural network techniques to capture local patterns in text effectively.
- **Training on Labeled Data**: Built on a dataset labeled with sentiments, enabling the model to learn and predict sentiments accurately.
- **Performance Metrics**: Model performance evaluated using accuracy, precision, recall, and F1-score.

## Getting Started
### Prerequisites
- Python 3.x
- Required Python packages: PyTorch, Streamlit, Tokenizers

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MEdan-US/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```
2. Create the virtual environment
   ```bash
   conda create --name text_env -y
   conda activate text_env
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the sentiment analysis system:

1. Run the main script:
```
python -m streamlit run app.py
```
2. Follow the prompts to input vietnamese sentence.

## Acknowledgments
- Thanks to the original authors of the TextCNN model for their significant contributions to text classification.
- We appreciate the open-source community for providing various libraries and frameworks that made this project possible.

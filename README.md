# Vietnamese Sentiment Analysis Project

This project aims to perform sentiment analysis on Vietnamese text using the TextCNN model. TextCNN is a convolutional neural network model specifically designed for sentence classification tasks, making it suitable for analyzing sentiments based on Vietnamese language texts.

## Features
- **Vietnamese Language Processing**: Tailored for processing and analyzing Vietnamese text data.
- **TextCNN Architecture**: Utilizes convolutional neural network techniques to capture local patterns in text effectively.
- **Training on Labeled Data**: Built on a dataset labeled with sentiments, enabling the model to learn and predict sentiments accurately.
- **Performance Metrics**: Model performance evaluated using accuracy, precision, recall, and F1-score.

## Getting Started
### Prerequisites
- Python 3.x
- Required Python packages: TensorFlow, Keras, Numpy, Pandas

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MEdan-US/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
To train the TextCNN model, run:
```bash
python train.py
```
This script will process the data and train the model using the specified parameters.

### Evaluating the Model
To evaluate the model's performance on the test set, run:
```bash
python evaluate.py
```

### Predicting Sentiment
To use the model for predicting sentiment on new text inputs, run:
```bash
python predict.py "your text here"
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the original authors of the TextCNN model for their significant contributions to text classification.
- We appreciate the open-source community for providing various libraries and frameworks that made this project possible.
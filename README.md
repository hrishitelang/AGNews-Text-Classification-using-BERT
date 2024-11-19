# AG News Text Classification using BERT

## Business Objective

The most abundant data today is text, which often exists in an unstructured format. Extracting insights from text can be complex and time-consuming. The project leverages **BERT (Bidirectional Encoder Representations from Transformers)**, a state-of-the-art NLP model, to classify news articles into four categories:
- **World**
- **Sports**
- **Business**
- **Sci/Tech**

This project demonstrates how fine-tuning BERT on a large dataset like **AG News** can achieve highly accurate classification results.

---

## Aim

The aim is to build, train, and fine-tune the BERT model to classify text from the AG News dataset into its respective categories with high accuracy and efficiency.

---

## Data Description

- **Dataset**: AG News Corpus (from Hugging Face library)
- **Structure**:
  - Training samples: 120,000 (30,000 per class)
  - Testing samples: 7,600 (1,900 per class)
  - Features: 
    - **Text**: News article title and description
    - **Label**: Class of the news article (0: World, 1: Sports, 2: Business, 3: Sci/Tech)

For my project, I used AGnews as one of the datasets from the hugging face library. The BERT model will be built on the AG News dataset.

AG News (AG’s News Corpus) is a sub dataset of AG's corpus of news articles constructed by assembling titles and description fields of articles from the 4 largest classes. The four classes are: World, Sports, Business, Sci/Tech

The AG News contains 30,000 training and 1,900 test samples per class. 

---

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `ktrain` for training and fine-tuning the BERT model
  - `transformers` for BERT implementation
  - `datasets` for accessing the AG News dataset
  - `numpy`, `pandas` for data manipulation
  - `tensorflow` for deep learning
- **Environment**: Jupyter Notebook

---

## Approach

### Step 1: Data Setup
1. Import the AG News dataset from the Hugging Face library.
2. Split the dataset into training and testing subsets.
3. Convert the dataset into a DataFrame for compatibility with `ktrain`.

### Step 2: Data Preprocessing
1. **Text Standardization**:
   - Pad sequences shorter than 512 tokens.
   - Truncate sequences longer than 512 tokens to fit BERT's input requirements.
2. **Class Mapping**:
   - Map class labels (0, 1, 2, 3) to their corresponding categories.

### Step 3: Building and Training the BERT Model
1. Load a pre-trained BERT base model (`bert-base-uncased`).
2. Fine-tune the model using the **ktrain** library:
   - Batch size: 16
   - Learning rate: 2e-5
   - Epochs: 3
3. Use the **one-cycle learning rate policy** to optimize training.

### Step 4: Model Evaluation
1. Evaluate the model's performance on the test set using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
2. Confusion matrix visualization for deeper insight into misclassifications.

---

## Performance Metrics

- **Accuracy**: 95%
- **Class-wise Metrics**:
  - **World**: Precision 0.97, Recall 0.96
  - **Sports**: Precision 0.99, Recall 0.99
  - **Business**: Precision 0.93, Recall 0.92
  - **Sci/Tech**: Precision 0.92, Recall 0.94

---

## Challenges Faced

1. **Long Sequence Handling**:
   - Managed sequences exceeding BERT's 512-token limit by truncating or padding.
2. **Training Time**:
   - Fine-tuning a large model like BERT required significant computational resources.
3. **Class Imbalance**:
   - Ensured balanced training by splitting the dataset evenly across classes.

---

## Key Learnings

1. Understanding the architecture and training of **BERT**:
   - Special tokens like `[CLS]`, `[SEP]`
   - Tasks like **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**
2. The importance of preprocessing for NLP pipelines:
   - Tokenization
   - Padding and truncation
3. Fine-tuning pre-trained transformer models for custom classification tasks.
4. Effective evaluation using metrics like F1-score and confusion matrices.

---

## Future Scope

1. **Multi-Language Support**:
   - Extend the model to classify articles in multiple languages.
2. **Real-Time Predictions**:
   - Deploy the trained model as a web service using frameworks like Flask or FastAPI.
3. **Larger Models**:
   - Experiment with larger versions of BERT (e.g., `bert-large`) or newer models like GPT.
4. **Explainability**:
   - Use tools like SHAP or LIME to explain the model's predictions.

---

## How to Run the Project

### Setup
1. Clone the repository and navigate to the project directory.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt


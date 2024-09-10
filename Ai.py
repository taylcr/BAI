# Import necessary libraries
import PyPDF2
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
        return text

# Function to preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

# Main execution block
if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = 'Bible.pdf'

    # Extract and preprocess the text
    pdf_text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(pdf_text)

    # Initialize tokenizer and model from Hugging Face's Transformers
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Prepare data for training
    inputs = tokenizer(processed_text, return_tensors='pt', max_length=512, truncation=True)
    dataset = Dataset.from_dict({'input_ids': inputs['input_ids'], 'labels': inputs['input_ids']})

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # Directory for saving logs and model checkpoints
        num_train_epochs=3,      # Number of training epochs
        per_device_train_batch_size=2,  # Batch size per device during training
        logging_steps=100,       # How often to log loss
        save_steps=500,          # How often to save the model
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

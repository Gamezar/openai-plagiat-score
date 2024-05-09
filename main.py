from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
import os
import sys





class ScoreCalculator:
    def __init__(self):
        logging.set_verbosity_error()
        absolute_path = os.path.dirname(__file__)
        relative_path = "./roberta-base-openai-detector/"
        full_path = os.path.join(absolute_path, relative_path)
        tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(full_path, local_files_only=True)
        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    def classifyText(self, text):
        if text is not None and text != "":
            res = self.classifier(text, truncation=True, max_length=510)
            label = res[0]['label']
            score = res[0]['score']

            if label == 'Real':
                real_score = score*100
                fake_score = 100-real_score
            else:
                fake_score = score*100
                real_score = 100-fake_score

            return [real_score, fake_score]
        else:
            return [50, 50]

def main():
    if len(sys.argv) < 2:
        print("Provide a file")
        sys.exit(1)

    file_path = sys.argv[1]

    with open(file_path, 'r') as file:
        text = file.read()

    sc = ScoreCalculator()
    real_score, fake_score = sc.classifyText(text)
    print(f"Originality score: {real_score}. Chatgpt score: {fake_score}")

if __name__ == "__main__":
    main()

import sys
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
class Inference:
    def __init__(self, model_path, text_descriptions, image_url):
        self.model_path = model_path
        self.text_descriptions = text_descriptions
        self.image_url = image_url

    def run_inference(self):
        try:
            # Load pre-trained CLIP model and processor
            model = CLIPModel.from_pretrained(self.model_path)
            processor = CLIPProcessor.from_pretrained(self.model_path)

            # Download and open the image
            image = Image.open(requests.get(self.image_url, stream=True).raw)

            # Process inputs
            inputs = processor(text=self.text_descriptions, images=image, return_tensors="pt", padding=True)

            # Run inference
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get the probabilities tensor
            probabilities = probs[0].tolist()

            # Iterate over the labels and their corresponding probabilities
            for i, (label, prob) in enumerate(zip(self.text_descriptions, probabilities)):
                print(f"Probability of '{label}' is: {prob}")

            # Get the label with the maximum probability
            max_prob_index = probabilities.index(max(probabilities))
            max_prob_label = self.text_descriptions[max_prob_index]
            print(f"\nThe provided photo is of '{max_prob_label}' with maximum probability.")

        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    # Read command-line arguments
    model_path = sys.argv[1]
    text_descriptions = sys.argv[2].split(',')
    image_url = sys.argv[3]

    # Create Inference Class Object
    I1 = Inference(model_path, text_descriptions, image_url)
    I1.run_inference()

if __name__ == "__main__":
    main()

# you can run the script with the following command:
# python3 p.py openai/clip-vit-base-patch32 "table,chair,rabbit" http://images.cocodataset.org/val2017/000000039769.jpg


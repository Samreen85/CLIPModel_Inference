# CLIP Inference
# Overview
This Python script performs image classification using the CLIP (Contrastive Language-Image Pre-training) model from Hugging Face. It takes an image URL and a list of text descriptions as input, and outputs the probabilities for each description.

## Usage
### 1. Install Dependencies:
Ensure you have the required dependencies installed. You can install them using the following command:

     pip install pillow requests transformers

### 2. Run the Script:
Execute the script by running the following command in your terminal:

    python script_name.py model_path "text1,text2,text3" image_url

Replace script_name.py with the actual name of your Python script, model_path with the path to the CLIP model, "text1,text2,text3" with a comma-separated list of text descriptions, and image_url with the URL of the image you want to classify. text1,2,3 are possible classes(labels).

## Example

    python script_name.py openai/clip-vit-base-patch32 "table,chair,rabbit" https://example.com/image.jpg
## Output
The script will display the probability of each text description and identify the label with the maximum probability.
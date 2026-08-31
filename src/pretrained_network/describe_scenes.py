from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model =BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def annotate_iamge(image: Image) -> None:
    image.show()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    
annotate_iamge(Image.open("./data/horse.jpg"))
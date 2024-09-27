import torch
from transformers import AutoProcessor, AutoModel

class Scorer:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def get_feat(self, text=None, image=None):
        inputs = self.processor(text=text, images=image, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            if text is None:
                image_embeds = self.model.vision_model(inputs['pixel_values'].to(self.device))[1]
                return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            
            elif image is None:
                text_embeds = self.model.text_model(inputs["input_ids"].to(self.device))[1]
                return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            else:
                raise ValueError("Only provide text or image, not both.")
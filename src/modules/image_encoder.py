import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Tuple, Union

class ImageEncoder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14-336"):
        """Initialize the image encoder using CLIP.
        
        Args:
            clip_model_name: HuggingFace model name for CLIP
        """
        super().__init__()
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Add projection layers for valence and arousal
        hidden_dim = self.clip_model.config.projection_dim
        projection_dim = hidden_dim // 2
        
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, images: Union[Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get valence and arousal predictions.
        
        Args:
            images: Either PIL images or tensors in CLIP format
            
        Returns:
            Tuple of predicted valence and arousal scores
        """
        # Process images if they're PIL images
        if isinstance(images, Image.Image):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            pixel_values = images.to(self.device)
            
        # Get CLIP image features
        image_features = self.clip_model.get_image_features(pixel_values)
        
        # Project to valence and arousal scores
        valence = self.valence_head(image_features)
        arousal = self.arousal_head(image_features)
        
        return valence, arousal
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Get the raw CLIP image embeddings.
        
        Args:
            image: PIL image to encode
            
        Returns:
            Image embedding tensor
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(inputs.pixel_values.to(self.device))
        return image_features
import torch
import os
from PIL import Image
import numpy as np
import datetime

from src.modules.image_encoder import ImageEncoder

# Add MIDI emotion model path to Python path
import sys
MIDI_EMOTION_PATH = os.path.join(os.path.dirname(__file__), "..", "midi_emotion", "src")
sys.path.append(MIDI_EMOTION_PATH)

class ARIA:
    """ARIA model that generates music from images based on emotional content."""
    
    def __init__(
        self,
        image_model_checkpoint: str,
        midi_model_dir: str,
        conditioning: str = "continuous_concat",
        device: str = None
    ):
        """Initialize ARIA model.
        
        Args:
            image_model_checkpoint: Path to image emotion model checkpoint
            midi_model_dir: Path to midi emotion model directory
            conditioning: Type of emotion conditioning to use. Options:
                - "continuous_concat": (Default) Embeds emotions as continuous vectors concatenated to all tokens
                - "continuous_token": Embeds emotions as continuous vectors prepended to the sequence
                - "discrete_token": Uses discrete emotion tokens (quantizes emotions into bins)
                - "none": No emotional conditioning
            device: Device to run on (default: auto-detect)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and not device == "cpu" else "cpu")
        self.conditioning = conditioning
        
        # Load image emotion model
        self.image_model = ImageEncoder()
        checkpoint = torch.load(image_model_checkpoint, map_location=self.device, weights_only=True)
        self.image_model.load_state_dict(checkpoint["model_state_dict"])
        self.image_model.eval()
        
        # Import midi generation
        from src.models.midi_emotion.src.generate import generate
        from src.models.midi_emotion.src.models.build_model import build_model
        self.generate_midi = generate
        
        # Load midi model
        model_fp = os.path.join(midi_model_dir, 'model.pt')
        mappings_fp = os.path.join(midi_model_dir, 'mappings.pt')
        config_fp = os.path.join(midi_model_dir, 'model_config.pt')
        
        self.maps = torch.load(mappings_fp, weights_only=True)
        config = torch.load(config_fp, weights_only=True)
        self.midi_model, _ = build_model(None, load_config_dict=config)
        self.midi_model = self.midi_model.to(self.device)
        self.midi_model.load_state_dict(torch.load(model_fp, map_location=self.device, weights_only=True))
        self.midi_model.eval()
    
    def generate(
        self,
        image_path: str,
        out_dir: str = "output",
        gen_len: int = 2048,
        temperature: list = [1.2, 1.2],
        top_k: int = -1,
        top_p: float = 0.7,
        min_instruments: int = 2
    ) -> tuple[float, float, str]:
        """Generate music from an image.
        
        Args:
            image_path: Path to input image
            out_dir: Directory to save generated MIDI
            gen_len: Length of generation in tokens
            temperature: Temperature for sampling [note_temp, rest_temp]
            top_k: Top-k sampling (-1 to disable)
            top_p: Top-p sampling threshold
            min_instruments: Minimum number of instruments required
            
        Returns:
            Tuple of (valence, arousal, midi_path)
        """
        # Get emotion from image
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            valence, arousal = self.image_model(image)
            valence = valence.squeeze().cpu().item()
            arousal = arousal.squeeze().cpu().item()
        
        # Create output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate MIDI
        continuous_conditions = np.array([[valence, arousal]], dtype=np.float32)
        
        # Generate timestamp for filename (for reference)
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        
        # Generate the MIDI
        self.generate_midi(
            model=self.midi_model,
            maps=self.maps,
            device=self.device,
            out_dir=out_dir,
            conditioning=self.conditioning,
            continuous_conditions=continuous_conditions,
            gen_len=gen_len,
            temperatures=temperature,
            top_k=top_k,
            top_p=top_p,
            min_n_instruments=min_instruments
        )
        
        # Find the most recently generated MIDI file
        midi_files = [f for f in os.listdir(out_dir) if f.endswith('.mid')]
        if midi_files:
            # Sort by creation time and get most recent
            midi_path = os.path.join(out_dir, max(midi_files, key=lambda f: os.path.getctime(os.path.join(out_dir, f))))
            return valence, arousal, midi_path
            
        raise RuntimeError("Failed to generate MIDI file")

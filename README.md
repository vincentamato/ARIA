# ARIA: Artistic Rendering of Images into Audio

An AI model that generates MIDI music based on the emotional content of artwork using CLIP-based image encoding.

[![GitHub](https://img.shields.io/badge/GitHub-ARIA-blue?logo=github)](https://github.com/vincentamato/aria)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ARIA-yellow)](https://huggingface.co/vincentamato/ARIA)

## Dependencies

This project uses the following external dependencies:
- [midi-emotion](https://github.com/serkansulun/midi-emotion) - For MIDI music generation based on emotions

## Setup

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/vincentamato/aria.git
cd aria
```

2. Initialize and update submodules (if cloned without --recursive):
```bash
git submodule init
git submodule update
```

3. Install the package and dependencies:
```bash
pip install -e .  # This will also install the local midi-emotion package
```

4. Download model files:
   - Visit [ARIA on Hugging Face](https://huggingface.co/vincentamato/ARIA)
   - Download the following files and place them in the corresponding directories:
     ```
     models/
     ├── continuous_concat/     # For continuous vector concatenation
     │   ├── model.pt
     │   ├── mappings.pt
     │   └── model_config.pt
     ├── continuous_token/      # For continuous vector prepending
     │   ├── model.pt
     │   ├── mappings.pt
     │   └── model_config.pt
     └── discrete_token/        # For discrete emotion tokens
         ├── model.pt
         ├── mappings.pt
         └── model_config.pt
     ```
   - Also download `image_encoder.pt` for the CLIP-based image emotion model

## How It Works

ARIA uses two main components:
1. A CLIP-based image encoder that extracts emotional content (valence and arousal) from artwork
2. A music generation model (midi-emotion) that creates MIDI music based on these emotions

### Emotion Conditioning Modes

ARIA supports different ways of incorporating emotional information into the music generation process:

- **Continuous Concat** (Default): Embeds emotions as continuous vectors and concatenates them to all tokens in the sequence. This provides consistent emotional influence throughout the generation.

- **Continuous Token**: Embeds emotions as continuous vectors and prepends them to the sequence. The emotional information is provided at the start of generation.

- **Discrete Token**: Quantizes emotions into discrete bins and uses them as special tokens. Useful when you want more distinct emotional categories.

- **None**: Generates music without emotional conditioning. Use this for baseline comparison or when you want purely structural music generation.

## Usage

Generate music from an image using the following command:

```bash
python src/models/aria/generate.py \
    --image path/to/your/image.jpg \
    --image_model_checkpoint path/to/image/model.pt \
    --midi_model_dir path/to/midi_emotion/model \
    --conditioning continuous_token \
    --out_dir output
```

### Required Arguments
- `--image`: Path to the input image
- `--image_model_checkpoint`: Path to the CLIP-based image emotion model checkpoint
- `--midi_model_dir`: Path to the midi-emotion model directory
- `--conditioning`: Type of emotion conditioning (choices: none, discrete_token, continuous_token, continuous_concat)

### Optional Arguments
- `--out_dir`: Directory to save generated MIDI (default: "output")
- `--gen_len`: Length of generation in tokens (default: 512)
- `--temperature`: Temperature for sampling [note_temp, rest_temp] (default: [1.2, 1.2])
- `--top_k`: Top-k sampling, -1 to disable (default: -1)
- `--top_p`: Top-p sampling threshold (default: 0.7)
- `--min_instruments`: Minimum number of instruments required (default: 1)
- `--cpu`: Force CPU inference
- `--batch_size`: Number of samples to generate (default: 1)

The model will output:
1. Predicted emotional values (valence and arousal)
2. Path to the generated MIDI file

## Attribution

This project incorporates the following open-source works:
- midi-emotion by Serkan Sulun et al. (https://github.com/serkansulun/midi-emotion) 
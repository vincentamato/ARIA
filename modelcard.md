---
license: mit
tags:
- art
- music
- midi
- emotion
- clip
- multimodal
---

# ARIA - Artistic Rendering of Images into Audio

ARIA is a multimodal AI model that generates MIDI music based on the emotional content of artwork. It uses a CLIP-based image encoder to extract emotional valence and arousal from images, then generates emotionally appropriate music using conditional MIDI generation.

## Model Description

- **Developed by:** Vincent Amato
- **Model type:** Multimodal (Image-to-MIDI) Generation
- **Language(s):** English
- **License:** MIT
- **Parent Model:** Uses CLIP for image encoding and midi-emotion for music generation
- **Repository:** [GitHub](https://github.com/vincentamato/aria)

### Model Architecture

ARIA consists of two main components:
1. A CLIP-based image encoder fine-tuned to predict emotional valence and arousal from images
2. A transformer-based MIDI generation model (midi-emotion) that conditions on these emotional values

The model offers three different conditioning modes:
- `continuous_concat`: Emotions as continuous vectors concatenated to all tokens
- `continuous_token`: Emotions as continuous vectors prepended to sequence
- `discrete_token`: Emotions quantized into discrete tokens

### Usage

The repository contains three variants of the MIDI generation model, each trained with a different conditioning strategy. Each variant includes:
- `model.pt`: The trained model weights
- `mappings.pt`: Token mappings for MIDI generation
- `model_config.pt`: Model configuration

Additionally, `image_encoder.pt` contains the CLIP-based image emotion encoder.

## Intended Use

This model is designed for:
- Generating music that matches the emotional content of artwork
- Exploring emotional transfer between visual and musical domains
- Creative applications in art and music generation

### Limitations

- Music generation quality depends on the emotional interpretation of input images
- Generated MIDI may require human curation for professional use
- Model's emotional understanding is limited to valence-arousal space

## Training Data

The model combines:
1. Image encoder: Fine-tuned on a curated dataset of artwork with emotional annotations
2. MIDI generation: Uses the Lakh-Spotify dataset as processed by the midi-emotion project

## Attribution

This project builds upon:
- **midi-emotion** by Serkan Sulun et al. ([GitHub](https://github.com/serkansulun/midi-emotion))
  - Paper: "Symbolic music generation conditioned on continuous-valued emotions" ([IEEE Access](https://ieeexplore.ieee.org/document/9762257))
  - Citation: S. Sulun, M. E. P. Davies and P. Viana, "Symbolic Music Generation Conditioned on Continuous-Valued Emotions," in IEEE Access, vol. 10, pp. 44617-44626, 2022
- **CLIP** by OpenAI for the base image encoder architecture

## License

This model is released under the MIT License. However, usage of the midi-emotion component should comply with its GPL-3.0 license. 
import argparse
from src.models.aria.aria import ARIA

def main():
    parser = argparse.ArgumentParser(description="Generate music from images based on emotional content")
    
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--image_model_checkpoint", type=str, required=True,
                        help="Path to image emotion model checkpoint")
    parser.add_argument("--midi_model_dir", type=str, required=True,
                        help="Path to midi emotion model directory")
    parser.add_argument("--out_dir", type=str, default="output",
                        help="Directory to save generated MIDI")
    parser.add_argument("--gen_len", type=int, default=512,
                        help="Length of generation in tokens")
    parser.add_argument("--temperature", type=float, nargs=2, default=[1.2, 1.2],
                        help="Temperature for sampling [note_temp, rest_temp]")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling (-1 to disable)")
    parser.add_argument("--top_p", type=float, default=0.7,
                        help="Top-p sampling threshold")
    parser.add_argument("--min_instruments", type=int, default=1,
                        help="Minimum number of instruments required")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    parser.add_argument("--conditioning", type=str, required=True,
                        choices=["none", "discrete_token", "continuous_token", "continuous_concat"],
                        help="Type of conditioning to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of samples to generate (not used for image input)")
    
    args = parser.parse_args()
    
    # Initialize model
    model = ARIA(
        image_model_checkpoint=args.image_model_checkpoint,
        midi_model_dir=args.midi_model_dir,
        conditioning=args.conditioning,
        device="cpu" if args.cpu else None
    )
    
    # Generate music
    valence, arousal, midi_path = model.generate(
        image_path=args.image,
        out_dir=args.out_dir,
        gen_len=args.gen_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_instruments=args.min_instruments
    )
    
    # Print results
    print(f"\nPredicted emotions:")
    print(f"Valence: {valence:.3f} (negative -> positive)")
    print(f"Arousal: {arousal:.3f} (calm -> excited)")
    print(f"\nGenerated MIDI saved to: {midi_path}")

if __name__ == "__main__":
    main()

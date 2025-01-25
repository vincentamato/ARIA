from setuptools import setup, find_packages

setup(
    name="aria",
    version="0.1.0",
    description="Generate music from artwork based on emotional content",
    author="Vincent Amato",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "Pillow",
        "numpy",
        "tqdm",
        "pretty_midi"
        "pypianoroll",
    ],
    dependency_links=[
        # Install local midi-emotion package
        f"file:///{__file__}/src/models/midi_emotion#egg=midi_emotion-0.1.0"
    ]
) 
from setuptools import setup, find_packages

setup(
    name="lie-detector",
    version="1.0.0",
    description="Voice-based lie/truth detection with trainable ML model and Gradio UI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "librosa>=0.10.0",
        "opensmile>=2.5.0",
        "scikit-learn>=1.3.0",
        "gradio>=4.19.0",
        "faster-whisper>=0.10.0",
        "plotly>=5.18.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "pydub>=0.25.1",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
    ],
    entry_points={
        "console_scripts": [
            "lie-detector=run:main",
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="spotify-recsys",
    version="0.1.0",
    description="Spotify Track Recommendation Engine - Content-based + Collaborative Filtering",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "spotipy>=2.19.0",
        "librosa>=0.9.0",
        "xgboost",
        "lightgbm",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "streamlit>=1.0.0",
        "mlflow>=1.20.0",
        "great-expectations>=0.13.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "jupyter>=1.0.0",
            "ipython>=7.25.0",
        ]
    },
)

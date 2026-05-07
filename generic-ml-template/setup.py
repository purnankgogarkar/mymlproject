from setuptools import setup, find_packages

setup(
    name="generic-ml-template",
    version="0.1.0",
    description="Generic ML pipeline framework for any tabular data (CSV/Excel) - supports classification, regression, and time-series",
    author="ML Template Framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.5.0",
        "optuna>=2.10.0",
        "streamlit>=1.0.0",
        "plotly>=5.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "bandit>=1.7.0",
        ]
    },
)

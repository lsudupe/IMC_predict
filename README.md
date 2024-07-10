# IMC Spatial predictions

# Breast Cancer Classification Using IDR Open Microscopy Dataset

## Overview
This project aims to classify breast cancer using images and tabular data from the IDR Open Microscopy database. Our primary goal is to detect the presence of Tertiary Lymphoid Structures (TLS) in breast cancer tissues.

## Data Source
The data is sourced from the IDR Open Microscopy database, accessible at [IDR Open Microscopy](https://idr.openmicroscopy.org/about.html). The specific dataset used is `idr0076`, which contains both images and tabular data relevant to our analysis.
Please refer to the `README.md` file in the `data` directory for more information on the dataset.

## Objective
Our objective is to leverage the dataset to predict the presence of TLS in breast cancer tissues, focusing on key features such as Celularity, NHG (Nottingham Histologic Grade), and LNEP (Lymph Node Examination Positive). The dataset includes markers CD68, CD3, and CD20, which are crucial for our analysis.

## Approach
We plan to use Graph Neural Networks (GNNs) for this classification task. GNNs are particularly suited for this project due to their ability to capture the complex relationships and patterns within the data.

## Dependency Management
This project uses [Poetry](https://python-poetry.org/) for dependency management. Poetry simplifies the management of project dependencies, ensuring that the project environment is reproducible and consistent across different setups. It uses the `pyproject.toml` file to declare project dependencies and manage virtual environments.

To install the dependencies for this project, run the following command:

```bash
poetry install
```

## Code Style

This project adheres to code style standards enforced by `black` and `flake8`.

### Black

`Black` is a Python code formatter that re-formats your code to make it more readable and 'pythonic'. To ensure your code is in line with the project's standards, please run `black` before committing your code. You can install and run `black` using the following commands:

### Flake8

`Flake8` is a Python linting tool that checks your code for potential errors and enforces a coding standard. To ensure your code is in line with the project's standards, please run `flake8` before committing your code. You can install and run `flake8` using the following commands:


## Inspiration
Our work is inspired by the "MULTIPLAI" GitHub repository, which has laid foundational work in the field of medical image analysis using AI. Additionally, we will reference existing Kaggle notebooks that have approached similar problems, adapting and improving upon their methodologies for our specific dataset.

## Getting Started
(Instructions on setting up the project locally, including environment setup, data download instructions, and a basic example of running the analysis)

## Usage
(A brief section on how to use the project, including running the models and interpreting results)

## Contributing
(Information for those who wish to contribute to the project, including how to submit pull requests, report bugs, or suggest enhancements)

## License
(Information about the project's license, if applicable)

## Acknowledgments
- MULTIPLAI GitHub Repository for inspiration and foundational concepts.
- Kaggle Community for notebooks and discussions that have guided the initial phase of this project.
- IDR Open Microscopy for providing the dataset essential for this research.

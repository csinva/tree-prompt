from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'datasets',
    'dict_hash',
    'imodels',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'torch',
    'tqdm',
    'transformers[torch] >= 4.23.1',
]

setuptools.setup(
    name="treeprompt",
    version="0.0.1",
    author="John X. Morris, Chandan Singh, Yuntian Deng, Sasha Rush",
    author_email="",
    description="Tree prompting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/tree-prompt",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

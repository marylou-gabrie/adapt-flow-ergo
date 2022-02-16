import setuptools


long_description="""Python package for JAMS described in
A Framework for Adaptive MCMC Targeting Multimodal Distributions
Emilia Pompe, Chris Holmes, Krzysztof Łatuszyński
"""

setuptools.setup(
    name="jams", # Replace with your own username
    version="0.0.1",
    author="Marylou",
    author_email="marylou.gabrie@polytechnique.edu",
    description="python package for JAMS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

import setuptools

long_description = """
Python package for adaptive sampling experiments, accompanying 
publication 
 Adaptation of the Independent Metropolis-Hastings Sampler with Normalizing Flow Proposals
 ----
"""

setuptools.setup(
    name="adaflonaco",
    version="0.0.1",
    author="----",
    author_email="---",
    description="python package for sampling with real-nvp flows",
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

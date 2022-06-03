import setuptools

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="diameter-clustering",
    version="0.1.0",
    author="Anton Klenitskiy",
    author_email="ant-klen@yandex.ru",
    description="Clustering with maximum distance between points inside clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antklen/diameter-clustering",
    packages=['diameter_clustering', 'diameter_clustering.approx'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    install_requires=[
        'hnswlib',
        'numpy',
        'numpy_groupies',
        'scikit_learn',
        'scipy',
        'tqdm'
    ],
)

from setuptools import setup, find_packages

setup(
    name='ml_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'timm',
        'pandas',
        'scikit-learn',
    ],
)

from setuptools import setup, find_packages

with open('requirements.txt', 'r', encoding='utf8') as fp:
    requirements = fp.read().split()

setup(
    name="transcriber",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'transcriber.server=transcriber.server:main',
        ],
    },
)
import io
from setuptools import setup
setup(
    name='tindetheus',
    version=open('tindetheus/VERSION').read().strip(),
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['tindetheus', 'tindetheus.facenet_clone',
              'tindetheus.facenet_clone.align'],
    package_data={'tindetheus': ['VERSION', '*.npy',
                                 'facenet_clone/align/*.npy']},
    entry_points={'console_scripts': ['tindetheus = tindetheus:command_line_run']},  # noqa E501
    url='https://github.com/cjekel/tindetheus',
    license='MIT License',
    description='Build personalized machine learning models for Tinder based on your historical preference',  # noqa E501
    long_description=io.open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    platforms=['any'],
    install_requires=[
        "numpy >= 1.11.3",
        "matplotlib >= 2.0.0",
        "imageio >= 2.2.0",
        "scikit-learn >= 0.19.0",
        "scikit-image >= 0.13.1",
        "tensorflow < 2.0.0",
        "pandas >= 0.21.0",
        "future >= 0.16.0",
        "joblib >= 0.13.0",
        "setuptools >= 38.6.0",
        ],
)

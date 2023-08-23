from setuptools import setup, find_packages

setup(
    name='tifascore',
    version='1.1.0',    
    description='TIFA: Text-to-Image Faithfulness Evaluation with Question Answering',
    url='https://github.com/Yushi-Hu/tifa',
    author='Yushi Hu',
    license='Apache License 2.0',
    packages=find_packages("tifascore"),
    install_requires=['numpy',
                      'torch',
                      'tqdm',
                      'word2number',
                      'Pillow',
                      'openai>=0.27.2',
                      'transformers>=4.31.0',
                      'fairseq',
                      'evaluate',
                      'salesforce-lavis',
                      'modelscope[multi-modal]',
                      'promptcap>=1.0.3',
                      ],

)

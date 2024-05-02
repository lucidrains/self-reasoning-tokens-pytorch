from setuptools import setup, find_packages

setup(
  name = 'self-reasoning-tokens-pytorch',
  packages = find_packages(exclude = []),
  version = '0.0.1',
  license='MIT',
  description = 'Self Reasoning Tokens',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/self-reasoning-tokens-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'adaptive computation'
  ],
  install_requires=[
    'einops>=0.8.0',
    'x-transformers>=1.28.4',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)

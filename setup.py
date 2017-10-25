from setuptools import setup, find_packages

setup(name='featrans',
      version='0.2',
      description='A light pacakges that achieves pmml goal between python and spark',
      url='http://github.com/SamuelDSR/featrans',
      author='SamuelDSR',
      author_email='samuel.longshihe@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['dill'],
      zip_safe=False)

from setuptools import find_packages, setup

setup(name='kosma_py_lib',
      version=open("VERSION").read(),
      description=('Python functions for the kosma gildas expansion'),
      include_package_data=True,
      packages=['kosma_py_lib'],
      package_data={'kosma_py_lib': ['templates/*']},
      url='',
      author='Christof Buchbender',
      author_email='buchbend@ph1.uni-koeln.de',
      license='MIT',
      install_requires=["pandas==0.22","jinja2","pyyaml", "sklearn", "astroML","pytz"],
      zip_safe=False)

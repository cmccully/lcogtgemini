from setuptools import setup

setup(name='lcogtgemini',
      author=['Curtis McCully'],
      author_email=['cmccully@lco.global'],
      version=0.1,
      py_moduldes=['reduce'],
      install_requires=['numpy', 'astropy', 'scipy'],
       entry_points={'console_scripts': ['reduce_gemini=reduce:run']})

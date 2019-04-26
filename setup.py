from setuptools import setup

setup(name='lcogtgemini',
      author=['Curtis McCully'],
      author_email=['cmccully@lco.global'],
      version=0.1,
      packages=['lcogtgemini'],
      install_requires=['numpy', 'astropy', 'scipy', 'matplotlib', 'pyraf', 'astroscrappy', 'statsmodels'],
      package_data={'': ['bpm_gn.fits', 'bpm_gs.fits', 'telluric_model.dat']},
      entry_points={'console_scripts': ['reduce_gemini=lcogtgemini.main:run']})

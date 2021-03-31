from setuptools import setup, find_packages

name = 'jexosim'
description = 'Time-domain simulator for JWST transit spectroscopy'
url = 'https://github.com/subisarkar/JexoSim/'
install_requires = ['pytransit==2.1.1', 'scipy==1.5.2', 'astropy==4.0.1', 'pandas==1.1.1', 'emcee==3.0.2', 'seaborn==0.10.1', 'uncertainties==3.1.4', 'tqdm==4.48.2', 'lxml==4.6.3']
entry_point = '__run__:console'
version = {'2.0'}


setup(
    name=name,
    version=version,
    description=description,
    url=url,
    author='Subhajit Sarkar',
    author_email='subhajit.sarkar@astro.cf.ac.uk',
    license='GPL',
    classifiers=[
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3'],
    entry_points={'console_scripts': ['{0} = {0}.{1}'.format(name, entry_point)]},
    packages=[name],
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)

 

from setuptools import setup


setup(
    name='pyberny',
    version='0.3.2',
    description='Molecular/crystal structure optimizer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jan Hermann',
    author_email='dev@janhermann.cz',
    url='https://github.com/azag0/pyberny',
    packages=['berny'],
    package_data={'berny': ['species-data.csv']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    install_requires=['numpy'],
    entry_points={
        'console_scripts': ['berny = berny.cli:main']
    },
)

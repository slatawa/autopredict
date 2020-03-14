import setuptools


with open('README.md') as f:

    README = f.read()



setuptools.setup(

    author="Sanchit Latawa",

    author_email="slatawa@yahoo.in",

    name='autopredict',

    license="MIT",

    description='autopredict is a package to automate Machine learning model selection tasks',

    version='v0.0.1',

    long_description=README,

    url='https://github.com/slatawa/autopredict.git',

    packages=setuptools.find_packages(),

    python_requires=">=3.5",

    install_requires=['requests'],

    classifiers=[

        # Trove classifiers

        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)

        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python',

        'Programming Language :: Python :: 3.5',

        'Programming Language :: Python :: 3.6',

        'Topic :: Software Development :: Libraries',

        'Topic :: Software Development :: Libraries :: Python Modules',

        'Intended Audience :: Developers',

    ],

)
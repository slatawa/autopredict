import setuptools


with open('README.md') as f:

    README = f.read()



setuptools.setup(

    author="Sanchit Latawa",

    author_email="slatawa@yahoo.in",

    name='autopredict',

    license="Apache Software License",

    description='Autopredict is a package to automate Machine learning model selection/ feature selection tasks',

    version='v1.0.5',

    long_description_content_type='text/markdown',

    long_description=README,

    url='https://github.com/slatawa/autopredict.git',

    packages=setuptools.find_packages(),

    python_requires=">=3.5",

    install_requires=['requests','pandas','numpy','sklearn'],

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
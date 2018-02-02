from distutils.core import setup
from os.path import join, dirname


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open(join(dirname(__file__), 'deepy3d/_version.py')) as versionpy:
    exec(versionpy.read())

with open('requirements.txt') as reqsfile:
    required = reqsfile.read().splitlines()

setup(
    name='deepy3d',
    version='0.1',
    description=("CT-scan bone segmentation using CNN's"),
    packages=['deepy3d'],
    install_requires=required
    url="https://github.com/NLeSC/yeap16-ai-3d-printing",
    license='Apache 2.0',
    long_description=open('README.txt').read(),
)

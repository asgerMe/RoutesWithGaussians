from setuptools import setup, find_packages

setup(
    name="steamrexroute",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'get_route=routeModule.bin.command_line:main',
        ],
    },
    # metadata to display on PyPI
    author="Asger Meldgaard",
    author_email="asger.meldgaard@gmail.com",
    description="Find optimal routes with Google OR-tools and gaussian processes",
    include_package_data=True,
    classifiers=[
        'License :: MIT'
    ],
    #install_requires = ['numpy', 'googlemaps', 'ortools', 'nested_lookup'],
	dependency_links=['https://github.com/asgerMe/SteamRexRoutes/tarball/master#egg=package-1.0']
)
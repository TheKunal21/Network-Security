
# The setup.py file is an essential part of packaging and 
# distributing Python projects. It is used by setuptools 
# (or distutils in older Python versions) to define the configuration
# of your project, such as its metadata, dependencies, and entry points.

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file 
            requirements = file.readlines()
            # process each line 
            for requirement in requirements:
                requirement = requirement.strip()
                # ignore empty lines, comments, and '-e .'
                if requirement and not requirement.startswith('#') and requirement != '-e .':
                    requirement_list.append(requirement)
       
    
    except FileNotFoundError:
        print('requirements.txt file not found')
        
    return requirement_list


setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='kunal saini',
    author_email='cryptocoffee01@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
    )
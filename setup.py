#import libraries
from setuptools import find_packages,setup
from typing import List

#Define the constant for the -e. requiement
HYPHEN_E_DOT_REQUIREMENT = '-e .'

#Function for get requirements
def get_requirements(file_path:str) -> List[str]:
#Create an empty list that will store the requirements
    requirements= []

    with open(file_path) as file_obj:
        for line in file_obj:
            requirements.append(line.replace("\n", ""))

        #Remove the -e
        if HYPHEN_E_DOT_REQUIREMENT in requirements:
            requirements.remove(HYPHEN_E_DOT_REQUIREMENT)

        #Return the list of requirements

        return requirements
    
setup(
    name='Household Electricity Bill Prediction',
    version='0.0.1',
    author="western",
    author_email="tdsl.thenuka@gmail,com",
    description="This model helps to predict the next month household electricity",
    url="https://github.com/dilmith456/Electricity-Bill-Prediction.git",
    license="MIT",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)
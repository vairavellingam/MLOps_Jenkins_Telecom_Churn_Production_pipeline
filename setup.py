from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-JENKINS-CHURN",
    version="1.0",
    author="Vel",
    description="Telecom Churn Prediction - MLOps Pipeline with Jenkins CI/CD",
    packages=find_packages(),
    install_requires=requirements,
)

"""Setup file for project"""


from setuptools import setup, find_packages

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name                            = "sd",
    version                         = "0.1.0",
    description                     = "Sustainable Developement Project 2025",
    long_description                = long_description,
    long_description_content_type   = "text/markdown",
    author                          = "Phuc Thanh Nguyen",
    author_email                    = "phucnguyenthanh.stu@gmail.com",
    url                             = "https://github.com/PhucInAI/2025_SustainableDevelopment",
    packages                        = find_packages(),
    install_requires                = [],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",           # Minimum Python version
)

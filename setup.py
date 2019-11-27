import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl_exp",
    version="0.0.1",
    author="Pitrified",
    author_email="pitrified.git@gmail.com",
    description="Some reinforcement learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pitrified/rl-exp",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=["gym", "gym-racer", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

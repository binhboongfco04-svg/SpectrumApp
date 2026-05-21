from setuptools import setup, find_packages

setup(
    name="SpectrumApp",
    version="1.0.0",
    author="Nguyen Van Binh",
    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        "kivy==2.3.0",
        "kivymd==1.2.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "kivy-garden.graph==0.4.0"
    ],

    package_data={
        "": [
            "*.png",
            "*.json"
        ]
    },

    zip_safe=False,
)
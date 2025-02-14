from setuptools import setup, find_packages

setup(
    name="is_image_document_ai",
    version="0.0.1",
    author="Andreas Reich",
    author_email="logophoman@gmail.com",
    description="A simple AI package for classifying images as documents or general images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/logophoman/is_image_document_ai",
    packages=find_packages(include=["is_image_document_ai", "is_image_document_ai.*"]),
    install_requires=[
        "torch",
        "torchvision",
        "requests",
        "pillow",
    ],
    package_data={
        "is_image_document_ai": [
            "mobilenet_document_vs_image.pth",
            "tinycnn_document_vs_image.pth"
        ]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "is_image_document-ai=is_image_document_ai.inference:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)

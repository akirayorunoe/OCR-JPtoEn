# How to Use PDF Translator
Currently work but with really low accuracy, change the config of --oem & --psm for your own sake

## Installation

First, install the required dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```
Setting up Paths
Before running the script, you need to change the input_pdf_path and output_pdf_path in your script (main.py) to your own file paths. Open a terminal and create a virtual environment using the following commands:

```bash
python3 -m venv venv
```
Activate the virtual environment:

```bash
source venv/bin/activate
```
Running the Translator
Now, you can run the PDF Translator script:

```bash
python3 main.py
```
Ensure that the paths in input_pdf_path and output_pdf_path are set to the correct locations for your PDF files.

Remember to deactivate the virtual environment when you're done:
```bash
deactivate
```


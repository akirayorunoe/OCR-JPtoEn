# How to Use PDF Translator

Translate scan PDF with plain image each page
Currently work but with really low accuracy, change the config of --oem & --psm for your own sake

## Installation

Go to the project

```bash
cd OCR-JPtoEn
```

Before running the script, you need to change the input_pdf_path and output_pdf_path in your script (main.py) to your own file paths. Open a terminal and create a virtual environment using the following commands:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
[Mac] source venv/bin/activate

[Window] venv\Scripts\activate.bat
```

First, install the required dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

Now, you can run the PDF Translator script:

```bash
python3 main.py
```

Remember to deactivate the virtual environment when you're done:
```bash
deactivate
```


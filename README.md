# How to Use PDF Translator

Translate scan PDF with plain image each page
Currently work but with really low accuracy, change the config of --oem & --psm for your own sake

## Installation

Go to the project

```bash
cd OCR-JPtoEn
```

Before running the script, you need to download + install [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) and change the input_pdf_path and output_pdf_path in your script (main.py) to your own file paths. Example:

```bash
    # Mac
    input_pdf_path = "/Users/innotech/Downloads/[106424].pdf"
    output_pdf_path = "/Users/innotech/Downloads/translated_pdfs/merged_translated.pdf"
    # Window
    input_pdf_path = "pdf\_106424_.pdf"
    output_pdf_path = "pdf\_106424_m.pdf"
```

Change tesseract cmd PATH in module/ImageTextExtractorEasyOCR (or ImageTextExtractor which use tesseract only). Example:

```bash
    # Mac
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    # Window
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
```

Open a terminal and create a virtual environment using the following commands:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
[Mac] source venv/bin/activate

[Window] venv\Scripts\activate
# Note: If you run into issue: Open powershell the type

#Set-ExecutionPolicy RemoteSigned

#[A] Yes to All

```

First, install the required dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

Now, you can run the PDF Translator script:

```bash
python main.py
```

Remember to deactivate the virtual environment when you're done:
```bash
deactivate
```


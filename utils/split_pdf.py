# Import the correct class names
from PyPDF2 import PdfFileReader, PdfFileWriter
import os

# Updated split_pdf function using PdfFileReader
def split_pdf(input_path, output_folder, max_size=10):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_path, 'rb') as file:
        pdf_reader = PdfFileReader(file)
        total_pages = pdf_reader.numPages

        for i in range(0, total_pages, max_size):
            pdf_writer = PdfFileWriter()
            output_file_path = os.path.join(output_folder, f'part_{i//max_size + 1}.pdf')

            for j in range(i, min(i+max_size, total_pages)):
                pdf_writer.addPage(pdf_reader.getPage(j))

            with open(output_file_path, 'wb') as output_file:
                pdf_writer.write(output_file)

if __name__ == "__main__":
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Get the parent directory of the script
    parent_directory = os.path.dirname(current_script_path)

    # Move two levels up to the parent directory of the 'utils' folder
    parent_of_parent_directory = os.path.dirname(parent_directory)

    # Specify the input PDF path
    input_pdf_path = "/Users/innotech/Downloads/[106424].pdf"

    # Specify the output folder path (in the parent directory)
    output_folder_path = os.path.join(parent_of_parent_directory, "split_pdf_over_10mb")

    # Specify the maximum part size
    max_part_size = 10  # in MB

    # Call the split_pdf function
    split_pdf(input_pdf_path, output_folder_path, max_size=max_part_size)

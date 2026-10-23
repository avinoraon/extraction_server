import subprocess

# After `pip install -e .`, "pdf2md" is on your PATH.
# If you didn't install it, point at the script directly:
CMD = ["python3", "main.py"]

# What you want to run:
subprocess.run(CMD + [
    "Sensirion_Datasheet_SFA40.pdf",        # input PDF
    "-o", "output.md",        # output markdown
    "--img-dir", "extracted_img",
    "--dpi", "150",
])
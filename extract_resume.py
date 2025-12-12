from pypdf import PdfReader

reader = PdfReader("content/Resumes/Vedaang_Chopra_Prof_CV.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n\n"

print(text)

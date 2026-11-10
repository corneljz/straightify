# Welcome to our Python Group Project!!
This is Notelift, for those of you that ever tried using camscanner, pdfscannerm or your in-built camera app, you will notice that it has a feature that can straighten your documents/photos even when you took it from a very bad angle.

Our project is basically trying to make that exact program using python!

# Getting Started
1. Before doing anything, it is recommended for one to create a python environment (venv or use conda) as it will help you organize better or python installations and versions.

Unfortunately I use conda (Miniforge3 specifically) so for those who use it here is how you can make one:
```bash
conda create --name "ENVNAME" python=3.11
conda activate ENVNAME
```

2. Let's start installing the modules/libraries you need, try running:
```bash
 pip install -r requirements.txt
```
Inside of that .txt file is all the modules/libraries you need to run this program. You can check it out inside, or go to QNA below to know what it does!

3. Download tesseract OCR engine software (You can try via chocolatey)

# QNA:
1.  How do I know what you are doing with the code? (For the nerds)
Well, we actually provided some explanations alongside our code so you can follow along and master it yourself (And of course because this is the course's requirement).

2. What does the modules in requirements.txt do?
opencv-python -> Image/video manipulation
numpy -> Handle the math
pytesseract -> Tesseract OCR engine


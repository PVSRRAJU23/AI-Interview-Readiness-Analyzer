import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

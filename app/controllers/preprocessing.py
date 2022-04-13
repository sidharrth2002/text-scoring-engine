import re

def remove_special_characters(text):
    '''
    Remove special characters from text
    '''
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[-+_^]*", "", text)
    text = ' '.join(text.split())
    print(text)
    return text
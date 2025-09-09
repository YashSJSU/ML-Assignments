import os

SECRET_KEY = 1
FILEPATH = '/Users/yash/Desktop/HMM/browncorpus.txt'

def readText():
    LIMIT = 50_000
    allText_parts = []
    count = 0

    with open(FILEPATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # take substring from column 15 onward
            for ch in line[15:].lower():
                if count >= LIMIT:
                    break
                if ch.isalpha():
                    allText_parts.append(ch)
                    count += 1

    allText = ''.join(allText_parts)
    return allText

def encrypt(text, key):
    key %= 26
    out = []
    for ch in text:
            base = ord('a')
            out.append(chr((ord(ch) - base + key) % 26 + base))
    return ''.join(out)




if __name__ == "__main__":
    text = readText()
    encryptedText = encrypt(text,SECRET_KEY)

    filename = "encryptedText.txt"
    with open(filename, 'w') as file:
        file.write(encryptedText)

    print(f"Text content saved to '{filename}' successfully.")




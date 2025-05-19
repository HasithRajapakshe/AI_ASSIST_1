import nltk

# Download punkt and wordnet
nltk.download('punkt')
nltk.download('wordnet')
# Try to download punkt_tab specifically
try:
    nltk.download('punkt_tab')
except:
    print("punkt_tab not available as a separate download, but punkt should work")

# Print download locations to verify
import os
print("\nNLTK data directories:")
for path in nltk.data.path:
    print(f"- {path}")
    if os.path.exists(path):
        print("  (exists)")
    else:
        print("  (does not exist)")

print("\nVerifying punkt tokenizer availability:")
try:
    from nltk.tokenize import word_tokenize
    print(word_tokenize("This is a test sentence."))
    print("Word tokenizer works!")
except Exception as e:
    print(f"Error: {e}")

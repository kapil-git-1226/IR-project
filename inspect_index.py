import os
import sys
import pickle

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index', 'inverted_index.pkl')

print("Loading your exact Inverted Index...")
with open(INDEX_PATH, 'rb') as f:
    index_obj = pickle.load(f)

# Pick a few interesting search terms that you might test later
interesting_words = ['jaguar', 'python', 'apple', 'market']

print("=" * 60)
print("🔍 PEEKING INSIDE THE INVERTED INDEX")
print("=" * 60)

for word in interesting_words:
    # Get the raw list from the dictionary
    postings = index_obj.index.get(word, [])
    
    print(f"\nDictionary Key : \"{word}\"")
    print(f"Total Documents: {len(postings)} articles")
    
    if postings:
        print("Data Structure : [ (Document ID, Times Word Appeared), ... ]")
        # Just show the first 5 so it doesn't flood the screen
        print(f"First 5 entries: {postings[:5]}")
        print("                 👆 Notice how each article counted the word!")
    else:
        print("This word never appeared in the 1987 Reuters news dataset!")
    
    print("-" * 60)

print(f"\nTotal unique words stored exactly like this: {len(index_obj.index):,}")

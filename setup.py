import pip
def main():
    pip.main(["install","-r","requirements.txt"])
    import nltk
    nltk.download("wordnet")
if __name__ == '__main__':
    main()

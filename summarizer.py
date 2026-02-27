import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text, sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary = summarizer(parser.document, sentences)
    return " ".join([str(sentence) for sentence in summary])


if __name__ == "__main__":
    article = input("Paste News Article:\n")

    print("\nSummary:")
    print(summarize_text(article))
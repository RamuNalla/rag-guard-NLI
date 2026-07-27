import spacy
from spacy.lang.en import English

class ClaimExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        # spaCy 3.8.x has a regex compilation bug on Python 3.14 when loading
        # en_core_web_sm (the infix rules use a regex construct that Python 3.14
        # rejects). We work around it by building a blank English pipeline and
        # adding only the sentencizer — which is all we ever needed anyway.
        try:
            self.nlp = spacy.load(model_name, exclude=["ner", "parser", "senter"])
        except Exception:
            # Fallback: bare English tokenizer + sentencizer (Python 3.14 safe)
            self.nlp = English()

        # Ensure we use the lightweight sentencizer
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def extract(self, text):
        """Splits a block of text into atomic sentences/claims."""
        doc = self.nlp(text)
        # Filter out very short strings that aren't real claim
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
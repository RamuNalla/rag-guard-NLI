import spacy

class ClaimExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        # We disable NER and the parser to make sentence splitting blazing fast
        self.nlp = spacy.load(model_name, exclude=["ner"])
        # Ensure we use the lightweight sentencizer
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def extract(self, text):
        """Splits a block of text into atomic sentences/claims."""
        doc = self.nlp(text)
        # Filter out very short strings that aren't real claims
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
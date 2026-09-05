import sys
import unicodedata

import torch


class InflectionalLexiconProcessor(object):
    def __init__(self, lexicon, vocab, pretrain, pos_lemma_pretag=False):
        # fills hypothesis_dictionary_xpos
        self.hypothesis_dictionary_xpos = {}
        self.hypothesis_dictionary_upos = {}
        self.hypothesis_dictionary_feats = {}
        # fallback for when xpos tag in influectional lexicon is not in vocab
        self.hypothesis_dictionary_xpos_fallback = {}
        self.hypothesis_dictionary_upos_fallback = {}
        self.hypothesis_dictionary_feats_fallback = {}

        closed_classes_xpos_rules = ['Z', 'punct'] if pos_lemma_pretag else []
        closed_classes_upos_rules = ['PUNCT', 'SYM'] if pos_lemma_pretag else []

        self.closed_classes_xpos = set()
        self.closed_classes_upos = set()

        self.xpos_vocab = vocab['xpos']
        self.upos_vocab = vocab['upos']
        self.feats_vocab = vocab['feats']

        self.create_closed_classes_upos(vocab['upos'], closed_classes_upos_rules)
        self.create_closed_classes_xpos(vocab['xpos'], closed_classes_xpos_rules)

        self.closed_classes_upos_inverse = [vocab['upos'][el] for el in vocab['upos'] if
                                       vocab['upos'][el] not in self.closed_classes_upos]
        self.closed_classes_xpos_inverse = [vocab['xpos'][el] for el in vocab['xpos'] if
                                       vocab['xpos'][el] not in self.closed_classes_xpos]

    def process_xpos(self, padded_prediction, word_strings):
        predictions = []

        padded_prediction = padded_prediction.detach().cpu().numpy()
        max_value = padded_prediction.argmax(2).tolist()

        for sent_id, (sent_indices, sent_strings) in enumerate(zip(max_value, word_strings)):
            sent_predictions = []
            for word_id, (word_prediction, word_string) in enumerate(zip(sent_indices, sent_strings)):
                if word_prediction not in self.closed_classes_xpos:
                    prediction = self.xpos_vocab[word_prediction]
                else:
                    prediction = self.xpos_vocab[self.closed_classes_xpos_inverse[
                        padded_prediction[sent_id, word_id, self.closed_classes_xpos_inverse].argmax()]]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def process_upos(self, padded_prediction, word_strings, xpos_preds):
        predictions = []

        padded_prediction = padded_prediction.detach().cpu().numpy()
        max_value = padded_prediction.argmax(2).tolist()

        for sent_id, (sent_indices, sent_strings) in enumerate(zip(max_value, word_strings)):
            sent_predictions = []
            for word_id, (word_prediction, word_string) in enumerate(zip(sent_indices, sent_strings)):
                if word_prediction not in self.closed_classes_upos:
                    prediction = self.upos_vocab[word_prediction]
                else:
                    prediction = self.upos_vocab[self.closed_classes_upos_inverse[
                        padded_prediction[sent_id, word_id, self.closed_classes_upos_inverse].argmax()]]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def process_feats(self, padded_prediction, word_strings, xpos_preds, upos_preds):
        predictions = []
        padded_prediction = torch.cat(padded_prediction, 2).detach().cpu().tolist()

        for sent_id, (sent_strings, sent_xpos, sent_upos) in enumerate(zip(word_strings, xpos_preds, upos_preds)):
            sent_predictions = []
            for word_id in range(len(sent_strings)):
                word_feat = padded_prediction[sent_id][word_id]
                prediction = self.feats_vocab[word_feat]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def create_closed_classes_xpos(self, vocab, closed_classes_rules):
        """ Fills a set of closed classes, that contains xpos ids that are not permitted. """
        for key in vocab:
            if key[0] in closed_classes_rules or key == 'punct' and key in closed_classes_rules:
                self.closed_classes_xpos.add(vocab[key])

    def create_closed_classes_upos(self, vocab, closed_classes_rules):
        """ Fills a set of closed classes, that contains xpos ids that are not permitted. """
        for key in vocab:
            if key in closed_classes_rules:
                self.closed_classes_upos.add(vocab[key])


class SloveneInflectionalLexiconProcessor(InflectionalLexiconProcessor):
    def __init__(self, lexicon, vocab, pretrain, pos_lemma_pretag=False):
        super(SloveneInflectionalLexiconProcessor, self).__init__(lexicon, vocab, pretrain, pos_lemma_pretag)
        closed_classes_xpos_rules = ['P', 'S', 'C', 'Q']
        closed_classes_upos_rules = ['PRON', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART']

        self.xpos_vocab = vocab['xpos']
        self.upos_vocab = vocab['upos']
        self.feats_vocab = vocab['feats']

        self.extract_lexicon_data(lexicon)

        # fills closed_classes
        # self.closed_classes = set()
        self.create_closed_classes_xpos(vocab['xpos'], closed_classes_xpos_rules)
        self.create_closed_classes_xpos(vocab['upos'], closed_classes_upos_rules)
        self.closed_classes_xpos_inverse = [vocab['xpos'][el] for el in vocab['xpos'] if vocab['xpos'][el] not in self.closed_classes_xpos]
        self.closed_classes_upos_inverse = [vocab['upos'][el] for el in vocab['upos'] if vocab['upos'][el] not in self.closed_classes_upos]

    def process_xpos(self, padded_prediction, word_strings):
        predictions = []

        padded_prediction = padded_prediction.detach().cpu().numpy()
        max_value = padded_prediction.argmax(2).tolist()

        for sent_id, (sent_indices, sent_strings) in enumerate(zip(max_value, word_strings)):
            sent_predictions = []
            for word_id, (word_prediction, word_string) in enumerate(zip(sent_indices, sent_strings)):
                if word_string in self.hypothesis_dictionary_xpos:
                    # if only one possible prediction in hypothesis dictionary, take it!
                    if len(self.hypothesis_dictionary_xpos[word_string]) == 1:
                        prediction = self.hypothesis_dictionary_xpos[word_string][0]
                    elif self.xpos_vocab[word_prediction] in self.hypothesis_dictionary_xpos[word_string]:
                        prediction = self.xpos_vocab[word_prediction]
                    else:
                        optional_indices = [self.xpos_vocab[el] for el in self.hypothesis_dictionary_xpos[word_string]]
                        prediction = self.xpos_vocab[optional_indices[padded_prediction[sent_id, word_id, optional_indices].argmax()]]
                elif word_string in self.hypothesis_dictionary_xpos_fallback:
                    prediction = self.hypothesis_dictionary_xpos_fallback[word_string][0]
                else:
                    if word_prediction not in self.closed_classes_xpos:
                        prediction = self.xpos_vocab[word_prediction]
                    else:
                        prediction = self.xpos_vocab[self.closed_classes_xpos_inverse[padded_prediction[sent_id, word_id, self.closed_classes_xpos_inverse].argmax()]]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def process_upos(self, padded_prediction, word_strings, upos_preds):
        predictions = []

        padded_prediction = padded_prediction.detach().cpu().numpy()
        max_value = padded_prediction.argmax(2).tolist()

        for sent_id, (sent_indices, sent_strings, sent_upos) in enumerate(zip(max_value, word_strings, upos_preds)):
            sent_predictions = []
            for word_id, (word_prediction, word_string, word_upos) in enumerate(zip(sent_indices, sent_strings, sent_upos)):
                key_tuple = (word_string, word_upos)
                if key_tuple in self.hypothesis_dictionary_upos:
                    # if only one possible prediction in hypothesis dictionary, take it!
                    if len(self.hypothesis_dictionary_upos[key_tuple]) == 1:
                        prediction = self.hypothesis_dictionary_upos[key_tuple][0]
                    elif self.upos_vocab[word_prediction] in self.hypothesis_dictionary_upos[key_tuple]:
                        prediction = self.upos_vocab[word_prediction]
                    else:
                        optional_indices = [self.upos_vocab[el] for el in self.hypothesis_dictionary_upos[key_tuple]]
                        prediction = self.upos_vocab[optional_indices[padded_prediction[sent_id, word_id, optional_indices].argmax()]]
                elif key_tuple in self.hypothesis_dictionary_upos_fallback:
                    prediction = self.hypothesis_dictionary_upos_fallback[key_tuple][0]
                else:
                    # prediction = self.upos_vocab[padded_prediction[sent_id, word_id].argmax().item()]
                    if word_prediction not in self.closed_classes_upos:
                        prediction = self.upos_vocab[word_prediction]
                    else:
                        prediction = self.upos_vocab[self.closed_classes_upos_inverse[
                            padded_prediction[sent_id, word_id, self.closed_classes_upos_inverse].argmax()]]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def process_feats(self, padded_prediction, word_strings, xpos_preds, upos_preds):
        predictions = []
        padded_prediction = torch.cat(padded_prediction, 2).detach().cpu().tolist()

        for sent_id, (sent_strings, sent_xpos, sent_upos) in enumerate(zip(word_strings, xpos_preds, upos_preds)):
            sent_predictions = []
            for word_id, (word_string, word_xpos, word_upos) in enumerate(zip(sent_strings, sent_xpos, sent_upos)):
                key_tuple = (word_string, word_xpos, word_upos)
                if key_tuple in self.hypothesis_dictionary_feats:
                    prediction = self.hypothesis_dictionary_feats[key_tuple][0]
                else:
                    word_feat = padded_prediction[sent_id][word_id]
                    prediction = self.feats_vocab[word_feat]
                sent_predictions.append(prediction)
            predictions.append(sent_predictions)
        return predictions

    def convert_feats(self, feats_string):
        return feats_string.replace(' ', '|')

    def extract_lexicon_data(self, lexicon):
        """ Creates hypothesis dictionary from lexicon. """
        if lexicon is None:
            raise Exception("Inflectional lexicon is only supported for Slovenian standard models. If you are using other models, set `pos_use_lexicon` to `False` (or remove it). Otherwise, you have to re-download Slovenian models. You can do this by using the following command: classla.download('sl')")
        for key in lexicon:
            if key[1] in self.xpos_vocab:
                self.hypothesis_dictionary_xpos.setdefault(key[0].lower(), []).append(key[1])
            else:
                self.hypothesis_dictionary_xpos_fallback.setdefault(key[0].lower(), []).append(key[1])

            if key[2] in self.upos_vocab:
                self.hypothesis_dictionary_upos.setdefault((key[0].lower(), key[1]), []).append(key[2])
            else:
                self.hypothesis_dictionary_upos_fallback.setdefault((key[0].lower(), key[1]), []).append(key[2])

            feats = self.convert_feats(key[3]) if key[3] else '_'
            self.hypothesis_dictionary_feats.setdefault((key[0].lower(), key[1], key[2]), []).append(feats)


processors = {"ssj": SloveneInflectionalLexiconProcessor, "sl_ssj": SloveneInflectionalLexiconProcessor}


class InflectionalLexicon:
    def __init__(self, lexicon, shorthand, vocab, pretrain, pos_lemma_pretag=False):
        """Base class for data converters for sequence classification data sets."""
        self.shorthand = shorthand
        assert shorthand in processors, f"Tag {shorthand} is not supported by inflectional lexicon."
        self.processor = processors[shorthand](lexicon, vocab, pretrain, pos_lemma_pretag)
        self.default_processor = DefaultPostprocessor(lexicon, vocab, pretrain, pos_lemma_pretag)

    def process_xpos(self, padded_prediction, word_strings):
        return self.processor.process_xpos(padded_prediction, word_strings)

    def process_upos(self, padded_prediction, word_strings, xpos_preds):
        return self.processor.process_upos(padded_prediction, word_strings, xpos_preds)

    def process_feats(self, padded_prediction, word_strings, xpos_preds, upos_preds):
        return self.processor.process_feats(padded_prediction, word_strings, xpos_preds, upos_preds)


class DefaultPostprocessor(InflectionalLexiconProcessor):
    def __init__(self, lexicon, vocab, pretrain, pos_lemma_pretag=False):
        super(DefaultPostprocessor, self).__init__(lexicon, vocab, pretrain, pos_lemma_pretag)

    def process_xpos(self, padded_prediction, word_strings):
        return super(DefaultPostprocessor, self).process_xpos(padded_prediction, word_strings)

    def process_upos(self, padded_prediction, word_strings, xpos_preds):
        return super(DefaultPostprocessor, self).process_upos(padded_prediction, word_strings, xpos_preds)

    def process_feats(self, padded_prediction, word_strings, xpos_preds, upos_preds):
        return super(DefaultPostprocessor, self).process_feats(padded_prediction, word_strings, xpos_preds, upos_preds)

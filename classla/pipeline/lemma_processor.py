"""
Processor for performing lemmatization
"""

from classla.models.common import doc
from classla.models.lemma.data import DataLoader
from classla.models.lemma.trainer import Trainer
from classla.pipeline._constants import *
from classla.pipeline.processor import UDProcessor, register_processor

@register_processor(name=LEMMA)
class LemmaProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([LEMMA])
    # set of processor requirements for this processor
    # pos will be added later for non-identity lemmatizerx
    REQUIRES_DEFAULT = set([TOKENIZE])
    # default batch size
    DEFAULT_BATCH_SIZE = 5000

    def __init__(self, config, pipeline, use_gpu):
        # run lemmatizer in identity mode
        self._use_identity = None
        super().__init__(config, pipeline, use_gpu)

    @property
    def use_identity(self):
        return self._use_identity

    def _set_up_model(self, config, use_gpu):
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self._config = config
            self.config['batch_size'] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            self._use_identity = False
            # add pos_model_path for inf. lexicon load and pos_lemma_pretag to see wether lemmas are transfered or not
            pos_processor = self.pipeline.processors['pos']
            pos_postprocessor = pos_processor.trainer.postprocessor
            inflectional_lexicon = getattr(pos_postprocessor, 'lemma_dictionary', None)
            inflectional_lexicon_entries = pos_processor.trainer.dict
            self._trainer = Trainer(
                args={
                    'pos_model_path': self.pipeline.config['pos_model_path'],
                    'pos_lemma_pretag': pos_processor.config['lemma_pretag'],
                },
                model_file=config['model_path'],
                use_cuda=use_gpu,
                inflectional_lexicon=inflectional_lexicon,
                inflectional_lexicon_entries=inflectional_lexicon_entries,
            )
            pos_processor.trainer.dict = None

    def _set_up_requires(self):
        if self.config.get('pos') and not self.use_identity:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, document):
        if not self.use_identity:
            batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        else:
            batch = DataLoader(document, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.doc.get([doc.TEXT, doc.XPOS]))
        else:
            if self.config.get('ensemble_dict', False):
                # skip the seq2seq model when we can
                skip = self.trainer.skip_seq2seq([(e[0].lower(),e[1],e[2]) for e in batch.doc.get([doc.TEXT, doc.XPOS, doc.LEMMA])])
                seq2seq_batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab,
                                           evaluation=True, skip=skip)
            else:
                seq2seq_batch = batch

            preds = []
            edits = []
            for i, b in enumerate(seq2seq_batch):
                ps, es = self.trainer.predict(b, self.config['beam_size'])
                preds += ps
                if es is not None:
                    edits += es

            if self.config.get('ensemble_dict', False):
                preds = self.trainer.postprocess([x for x, y in zip(batch.doc.get([doc.TEXT]), skip) if not y], preds, edits=edits)
                # expand seq2seq predictions to the same size as all words
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append('')
                    else:
                        preds1.append(preds[i])
                        i += 1
                preds = self.trainer.ensemble([(e[0].lower(),e[1],e[2]) for e in batch.doc.get([doc.TEXT, doc.XPOS, doc.LEMMA])], preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc

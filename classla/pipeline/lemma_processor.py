"""
Processor for performing lemmatization
"""

from collections import OrderedDict

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
    DEFAULT_CACHE_SIZE = 100000

    def __init__(self, config, pipeline, use_gpu):
        # run lemmatizer in identity mode
        self._use_identity = None
        self._prediction_cache = None
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
            cache_size = int(config.get('cache_size', self.DEFAULT_CACHE_SIZE))
            if cache_size < 0:
                raise ValueError("Lemma cache size cannot be negative.")
            self._prediction_cache = _LemmaPredictionCache(cache_size)
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
            batch = DataLoader(
                document, self.config['batch_size'], self.config, vocab=self.vocab,
                evaluation=True, inference_only=True)
        else:
            batch = DataLoader(document, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.doc.get([doc.TEXT, doc.XPOS]))
        else:
            if self.config.get('ensemble_dict', False):
                # skip the seq2seq model when we can
                entries = batch.doc.get([doc.TEXT, doc.XPOS, doc.LEMMA])
                pairs = [(e[0].lower(), e[1], e[2]) for e in entries]
                cache_keys = [(e[0], e[1]) for e in entries]
                skip = self.trainer.skip_seq2seq(pairs)
                cached = {}
                duplicate_of = {}
                pending = {}
                for index, (dictionary_hit, cache_key) in enumerate(zip(skip, cache_keys)):
                    if dictionary_hit:
                        continue
                    cached_prediction = self._prediction_cache.get(cache_key)
                    if cached_prediction is not None:
                        cached[index] = cached_prediction
                        skip[index] = True
                    elif cache_key in pending:
                        duplicate_of[index] = pending[cache_key]
                        skip[index] = True
                    else:
                        pending[cache_key] = index
                seq2seq_batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab,
                                           evaluation=True, skip=skip, inference_only=True)
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
                for index, s in enumerate(skip):
                    if index in cached:
                        preds1.append(cached[index])
                    elif index in duplicate_of:
                        preds1.append(preds1[duplicate_of[index]])
                    elif s:
                        preds1.append('')
                    else:
                        prediction = preds[i]
                        preds1.append(prediction)
                        self._prediction_cache.put(cache_keys[index], prediction)
                        i += 1
                preds = self.trainer.ensemble(pairs, preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc


class _LemmaPredictionCache:
    """A small LRU cache for deterministic seq2seq lemma predictions."""

    def __init__(self, max_size):
        self.max_size = max_size
        self._values = OrderedDict()

    def get(self, key):
        if self.max_size == 0:
            return None
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key, value):
        if self.max_size == 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > self.max_size:
            self._values.popitem(last=False)

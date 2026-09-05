"""
A trainer class to handle training and testing of models.
"""

import os
import pickle
from pathlib import Path
import sys
import time
import logging
import torch
from torch import nn

from classla.models.common.trainer import Trainer as BaseTrainer
from classla.models.common import utils, loss
from classla.models.pos.model import Tagger
from classla.models.pos.postprocessor import InflectionalLexicon, DefaultPostprocessor
from classla.models.pos.vocab import MultiVocab

logger = logging.getLogger('classla')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:8]]
    else:
        inputs = batch[:8]
    orig_idx = batch[8]
    word_orig_idx = batch[9]
    sentlens = batch[10]
    wordlens = batch[11]
    word_string = batch[12]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens, word_string


class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, share_hid=args['share_hid'])
            self.dict = None

        self.use_lexicon = args['use_lexicon'] if 'use_lexicon' in args else None
        self.pos_lemma_pretag = args['lemma_pretag'] if 'lemma_pretag' in args else None
        self.postprocessor = None
        if self.use_lexicon:
            args['shorthand'] = args['shorthand'] if 'shorthand' in args else self.args['shorthand']
            inflectional_lexicon = self.dict
            self.postprocessor = InflectionalLexicon(inflectional_lexicon, args['shorthand'], self.vocab, pretrain,
                                                     args['lemma_pretag'])
            # The postprocessor has extracted every view it needs, including
            # the lemma dictionary.  Do not retain the multi-million-entry raw
            # checkpoint list as a duplicate in memory.
            self.dict = None
        else:
            if self.pos_lemma_pretag:
                self.postprocessor = DefaultPostprocessor(None, self.vocab, None, pos_lemma_pretag=self.pos_lemma_pretag)
            else:
                self.postprocessor = None
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, word_string = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, word_string)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, word_string = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        with torch.inference_mode():
            _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, word_string, postprocessor=self.postprocessor, compute_loss=False)

        # upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
        # feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[2].tolist()]

        if self.postprocessor is None:
            upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
            xpos_seqs = [self.vocab['xpos'].unmap(sent) for sent in preds[1].tolist()]
            feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[2].tolist()]
        else:
            upos_seqs = preds[0]
            xpos_seqs = preds[1]
            feats_seqs = preds[2]

        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True, inflectional_lexicon=None):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }
        if inflectional_lexicon:
            params['dicts'] = inflectional_lexicon

        try:
            torch.save(params, filename)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, pretrain):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=False)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        emb_matrix = None
        if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
            emb_matrix = pretrain.emb
        self.model = Tagger(self.args, self.vocab, emb_matrix=emb_matrix, share_hid=self.args['share_hid'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.dict = checkpoint['dicts'] if 'dicts' in checkpoint else None

    @staticmethod
    def load_inflectional_lexicon(filename, entries=None):
        """Load a cached lemma view of the POS inflectional lexicon.

        The standard Slovene POS checkpoint contains millions of raw entries.
        Converting them on every pipeline startup is much slower than loading
        the model itself, so keep a versioned sidecar beside the checkpoint.
        """
        model_path = Path(filename)
        signature = (model_path.stat().st_size, model_path.stat().st_mtime_ns)
        cache_path = Path(str(model_path) + '.lemma_lexicon.pkl')

        cached = Trainer._read_inflectional_lexicon_cache(cache_path, signature)
        if cached is not None:
            return cached

        lock_path = Path(str(cache_path) + '.lock')
        owns_lock = False
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_fd)
            owns_lock = True
        except OSError:
            # Another worker may be producing the cache.  Wait for it rather
            # than making every process repeat the expensive conversion.
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                cached = Trainer._read_inflectional_lexicon_cache(cache_path, signature)
                if cached is not None:
                    return cached
                if not lock_path.exists():
                    break
                time.sleep(0.25)

        try:
            if entries is None:
                try:
                    checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=False)
                except BaseException:
                    logger.error("Cannot load model from {}".format(filename))
                    raise
                assert 'dicts' in checkpoint, Exception('Can not load inflectional dictionary. Make sure that your tagger model has it.')
                entries = checkpoint['dicts']

            # Reversing lets the dict comprehension retain the first original
            # entry while doing the large conversion in optimized C bytecode.
            inf_lexicon = {(entry[0], entry[1]): entry[4] for entry in reversed(entries)}
            Trainer._write_inflectional_lexicon_cache(cache_path, signature, inf_lexicon)
            return inf_lexicon
        finally:
            if owns_lock:
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _read_inflectional_lexicon_cache(cache_path, signature):
        try:
            with cache_path.open('rb') as stream:
                payload = pickle.load(stream)
            if payload.get('version') == 1 and payload.get('source') == signature:
                return payload.get('lexicon')
        except (OSError, EOFError, pickle.PickleError, AttributeError):
            pass
        return None

    @staticmethod
    def _write_inflectional_lexicon_cache(cache_path, signature, lexicon):
        temporary_path = Path(str(cache_path) + '.{}.tmp'.format(os.getpid()))
        try:
            with temporary_path.open('wb') as stream:
                pickle.dump(
                    {'version': 1, 'source': signature, 'lexicon': lexicon},
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary_path, cache_path)
        except OSError:
            logger.warning("Could not write inflectional lexicon cache to %s", cache_path)
            try:
                temporary_path.unlink()
            except OSError:
                pass

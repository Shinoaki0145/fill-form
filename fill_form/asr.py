"""Gipformer inference using the pinned upstream CUDA recipe."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import json
import numpy as np

MODEL_ID = 'g-group-ai-lab/gipformer1.5-68M-rnnt'

class Gipformer:
    def __init__(self, repo, icefall_dir, model_dir, decoding_method, beam_size):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Gipformer requires CUDA; CPU fallback is not allowed')
        self.torch = torch
        self.device = torch.device('cuda:0')
        sys.path.insert(0, str(repo))
        from infer_pytorch import setup_icefall, load_model_files
        setup_icefall(Path(icefall_dir))
        import k2
        import kaldifeat
        import sentencepiece as spm
        from beam_search import greedy_search_batch, modified_beam_search
        from torch.nn.utils.rnn import pad_sequence
        from train import add_model_arguments, get_model, get_params
        # Icefall sets inter-op once at import. Only intra-op can be adjusted here.
        torch.set_num_threads(2)
        self.greedy = greedy_search_batch
        self.beam_search = modified_beam_search
        self.pad_sequence = pad_sequence
        parser = argparse.ArgumentParser()
        parser.add_argument('--context-size', type=int, default=2)
        add_model_arguments(parser)
        old_git_dir = os.environ.get('GIT_DIR')
        try:
            os.environ['GIT_DIR'] = str(Path(icefall_dir)/'.git')
            params = get_params()
        finally:
            if old_git_dir is None:
                os.environ.pop('GIT_DIR', None)
            else:
                os.environ['GIT_DIR'] = old_git_dir
        params.update(vars(parser.parse_args([])))
        params.decoding_method, params.beam_size = decoding_method, beam_size
        paths = load_model_files(version='1.5', model_dir=str(model_dir))
        tokens = k2.SymbolTable.from_file(paths['tokens'])
        params.blank_id, params.unk_id = tokens['<blk>'], tokens['<unk>']
        count = sum(not symbol.startswith('#') for symbol in tokens.symbols)
        if params.blank_id == 0:
            count -= 1
        params.vocab_size = count+1  # Same vocabulary calculation as the official script.
        load_start = time.perf_counter()
        self.model = get_model(params)
        checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f'Checkpoint is missing model parameters: {incompatible.missing_keys[:20]}')
        self.model.to(self.device).eval()
        if not all(p.device.type == 'cuda' for p in self.model.parameters()):
            raise RuntimeError('Some Gipformer parameters are not on CUDA')
        torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter()-load_start
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(paths['bpe_model'])
        opts = kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self.fbank = kaldifeat.Fbank(opts)
        self.decoding_method, self.beam_size = decoding_method, beam_size
        self.metadata = dict(model=MODEL_ID, model_version='1.5', runtime='official PyTorch CUDA recipe',
            model_revision=Path(model_dir).name, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), k2_version=getattr(k2, '__version__', getattr(k2, '__dev_version__', None)),
            decoding_method=decoding_method, beam_size=beam_size if decoding_method=='modified_beam_search' else None,
            context_size=2, num_threads=torch.get_num_threads(),
            num_interop_threads=torch.get_num_interop_threads(), model_parameters=sum(p.numel() for p in self.model.parameters()),
            model_load_seconds=self.model_load_seconds,
            checkpoint_unexpected_keys=list(incompatible.unexpected_keys),
            cuda_checks=dict(model_parameters=True, features=False, encoder_output=False),
            repo_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
            icefall_commit=subprocess.check_output(['git','-C',str(icefall_dir),'rev-parse','HEAD'], text=True).strip())
        print(json.dumps(self.metadata, ensure_ascii=False, indent=2), flush=True)

    def recognize(self, wave):
        torch = self.torch
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Invalid waveform')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).to(self.device)
            features = self.fbank([tensor])
            lengths = torch.tensor([features[0].size(0)], device=self.device)
            features = self.pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
            if features.device.type != 'cuda':
                raise RuntimeError('Fbank features are not on CUDA')
            encoded, encoded_lengths = self.model.forward_encoder(features, lengths)
            if encoded.device.type != 'cuda':
                raise RuntimeError('Encoder output is not on CUDA')
            if self.decoding_method == 'greedy_search':
                hypotheses = self.greedy(model=self.model, encoder_out=encoded,
                                        encoder_out_lens=encoded_lengths)
            else:
                hypotheses = self.beam_search(model=self.model, encoder_out=encoded,
                                  encoder_out_lens=encoded_lengths, beam=self.beam_size)
            text = self.sp.decode(hypotheses[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        self.metadata['cuda_checks'].update(features=True, encoder_output=True)
        return text, elapsed


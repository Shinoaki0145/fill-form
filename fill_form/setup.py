"""Fetch pinned upstream inference sources and model; never change existing checkouts."""
import argparse
from pathlib import Path
import subprocess
import tomllib

from huggingface_hub import snapshot_download

GIP_COMMIT = '0cac5b183d2e1a9e3429ff69080e09cc62ebdf9b'
ICE_COMMIT = '1c388ad9be94c0ee9de76f7a6ed45b1cff677d13'
MODEL_REVISION = 'dd9227dcd8705c13f33bdbe59728d546ab94480f'


def fetch(url, directory, commit, sparse=()):
    directory = directory.resolve()
    if directory.exists():
        actual = subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'], text=True).strip()
        if actual != commit:
            raise RuntimeError(f'{directory} has another revision; use a new empty path')
        return
    directory.mkdir(parents=True)
    subprocess.run(['git','init',str(directory)], check=True)
    def git(*args):
        subprocess.run(['git','-C',str(directory), *args], check=True)
    git('remote','add','origin',url)
    git('fetch','--depth','1','--filter=blob:none','origin',commit)
    if sparse:
        git('sparse-checkout','init','--cone')
        git('sparse-checkout','set',*sparse)
    git('checkout','--detach','FETCH_HEAD')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config.toml'))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())['asr']
    base = args.config.resolve().parent
    fetch('https://github.com/ggroup-ai-lab/gipformer.git', base/config['repo'], GIP_COMMIT)
    fetch('https://github.com/k2-fsa/icefall.git', base/config['icefall'], ICE_COMMIT,
          ('icefall','egs/librispeech/ASR'))
    snapshot_download('g-group-ai-lab/gipformer1.5-68M-rnnt', revision=MODEL_REVISION,
                      allow_patterns=['model.pt','bpe.model','tokens.txt'], local_dir=base/config['model_dir'])
    print('Pinned Gipformer / Icefall / model ready.')


if __name__ == '__main__':
    main()

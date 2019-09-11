import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, blizzard2013
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'database/LJSpeech-1.0')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def preprocess_blizzard2013(args):
  in_dir = os.path.join(args.base_dir, 'database/blizzard2013/segmented')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard2013.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def write_metadata(agrs):
  out_dir = os.path.join(args.base_dir, agrs.output)
  with open(os.path.join(out_dir, agrs.save_txt), 'w', encoding='utf-8') as f:
    for ppgs in os.listdir(args.ppgs_dir):
      ppgs_name = ppgs.split('.npy')[0]
      mel = ppgs
      mel_path = os.path.join(args.mel_dir, mel)
      lpc32 = pps_name + '.mel.npy'
      lpc32_path = os.path.join(args.lpc32_dir, lpc32)
      if os.path.isfile(lpc32_path):
        if os.path.isfile(mel_path):
          spk=ppgs[:4]
          f.write('%s|%s|%s|%s\n'%(ppgs, mel, lpc32, spk))
      else:
        os.system('echo %s>>wrng.txt'%lpc32)
        print('%s is none'%lpc32_path)
    



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech', 'blizzard2013'])
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  write_meta(args)
  #if args.dataset == 'blizzard':
  #  preprocess_blizzard(args)
  #elif args.dataset == 'ljspeech':
  #  preprocess_ljspeech(args)
  #elif args.dataset == 'blizzard2013':
  #  preprocess_blizzard2013(args)


if __name__ == "__main__":
  main()

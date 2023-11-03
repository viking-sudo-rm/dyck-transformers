# TODO: Generate Dyck languages a la Suzgun
# TODO: Create GPT2 language model
# TODO: Train GPT2 on Dyck words

from argparse import ArgumentParser

from dyck import Dyck2Generator

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("-p", type=float, default=.5)
    parser.add_argument("-q", type=float, default=.25)

random.seed(42)
dyck_gen = Dyck2Generator(args.p, args.q, max_depth=100)

def word_in_length_range(dyck_gen, min_length, max_length):
    while True:
        word = dyck_gen.generate()
        if min_length <= len(word) <= max_length:
            return word

train = [word_in_length_range(dyck_gen, 2, 50) for _ in range(100000)]
test = [word_in_length_range(dyck_gen, 52, 1000) for _ in range(5000)]

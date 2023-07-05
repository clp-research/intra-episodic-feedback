import itertools
import json
import pickle
from typing import Dict

from cogrip.constants import COLORS, SHAPES, POSITIONS
from cogrip.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from cogrip.pentomino.symbolic.types import PropertyNames


def store_sentence_embeddings(sentences, embeddings, dir=".", max_sentence_length=11):
    fp = dir + '/sentence_embeddings.pkl'
    encodings = [encode_sent(s.lower(), max_sentence_length) for s in sentences]
    with open(fp, "wb") as fOut:
        pickle.dump({'sentences': sentences, 'code': encodings, 'embeddings': embeddings},
                    fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Stored embeddings to", fp)


def load_sentence_embeddings(file_path="./sentence_embeddings.pkl") -> Dict:
    """
    :return: {"sentences","code","embeddings"}
    """
    print("Loading embeddings from", file_path)
    with open(file_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data


def create_vocab():
    words = ["<pad>", "<s>", "<e>", "<unk>"]
    words += ["Take", "the", "piece", "at", "Select", "Get"]
    words += set([w for v in COLORS for w in v.value_name.split()])
    words += set([w for v in SHAPES for w in v.value.split()])
    words += set([w for v in POSITIONS for w in v.value.split()])
    words += ["Not", "this", "direction", "Yes", "there", "No", "Yeah", "way"]
    words = [w.lower() for w in words]
    print("Vocabulary size:", len(words))
    vocab = {
        "i2w": dict([(i, w) for i, w in enumerate(words)]),
        "w2i": dict([(w, i) for i, w in enumerate(words)]),
    }
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)


VOCAB = {"i2w": {"0": "<pad>", "1": "<s>", "2": "<e>", "3": "<unk>", "4": "take", "5": "the", "6": "piece", "7": "at",
                 "8": "select", "9": "get", "10": "cyan", "11": "green", "12": "red", "13": "brown", "14": "olive",
                 "15": "pink", "16": "navy", "17": "yellow", "18": "grey", "19": "purple", "20": "blue", "21": "orange",
                 "22": "t", "23": "l", "24": "i", "25": "u", "26": "x", "27": "z", "28": "v", "29": "y", "30": "w",
                 "31": "p", "32": "n", "33": "f", "34": "top", "35": "left", "36": "right", "37": "center",
                 "38": "bottom", "39": "not", "40": "this", "41": "direction", "42": "yes", "43": "there", "44": "no",
                 "45": "yeah", "46": "way"},
         "w2i": {"<pad>": 0, "<s>": 1, "<e>": 2, "<unk>": 3, "take": 4, "the": 5, "piece": 6, "at": 7, "select": 8,
                 "get": 9, "cyan": 10, "green": 11, "red": 12, "brown": 13, "olive": 14, "pink": 15, "navy": 16,
                 "yellow": 17, "grey": 18, "purple": 19, "blue": 20, "orange": 21, "t": 22, "l": 23, "i": 24, "u": 25,
                 "x": 26, "z": 27, "v": 28, "y": 29, "w": 30, "p": 31, "n": 32, "f": 33, "top": 34, "left": 35,
                 "right": 36, "center": 37, "bottom": 38, "not": 39, "this": 40, "direction": 41, "yes": 42,
                 "there": 43, "no": 44, "yeah": 45, "way": 46}}


def encode_sent(sent, pad_length=None):
    w2i = VOCAB["w2i"]
    if sent == "<silence>":
        tokens = []
    else:
        tokens = [w2i["<s>"]] + [w2i[w] for w in sent.split()] + [w2i["<e>"]]
    if pad_length:
        while len(tokens) < pad_length:  # make variable length sequences the same length
            tokens.append(w2i["<pad>"])
    return tokens


def decode_sent(tokens):
    i2w = VOCAB["i2w"]
    sent = []
    for t in tokens:
        if t < 4:
            continue
        sent.append(i2w[str(t)])
    return " ".join(sent)


def trans_obs(env, obs):
    if "mission" in obs:
        mission = decode_sent(obs["mission"])
    else:
        mission = "<no mission>"
    if "feedback" in obs:
        feedback = decode_sent(obs["feedback"])
    else:
        feedback = "<no feedback>"
    image = env.render(mode="rgb_array")
    return mission, feedback, image


def trans_obs_venv(venv, obs):
    n_envs = len(venv.envs)
    utterance = None
    if "language" in obs:
        utterance = [decode_sent(utt) for utt in obs["language"]]
    if utterance is None:
        utterance = ["<no  utterance>" for _ in range(n_envs)]
    images = venv.get_images()
    return utterance, images


def create_all_sents():
    colors = [None] + COLORS  # add option to not mention a property
    shapes = [None] + SHAPES  # add option to not mention a property
    positions = [None] + POSITIONS  # add option to not mention a property
    sentences = ["take the piece"]
    for start_token in ["take"]:
        reg = PentoIncrementalAlgorithm([PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION],
                                        start_tokens=[start_token])
        for color, shape, position in itertools.product(colors, shapes, positions):
            # print(color, shape, position)
            props = {}
            if color:
                props[PropertyNames.COLOR] = color
            if shape:
                props[PropertyNames.SHAPE] = shape
            if position:
                props[PropertyNames.REL_POSITION] = position
            sent = reg.verbalize_properties(props)
            sentences.append(sent)
        print(len(sentences), "...")
    negative_feedback = ["not this direction", "not there", "no", "not this piece"]
    sentences = sentences + negative_feedback
    positive_feedback = ["yes this direction", "yes", "yeah", "yes this way", "yes this piece"]
    sentences = sentences + positive_feedback
    sentences = sentences + ["<silence>"]  # empty / silence
    print(len(sentences))
    return sentences


if __name__ == '__main__':
    print(len(VOCAB["i2w"]))

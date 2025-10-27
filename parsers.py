from abc import ABC, abstractmethod
import json
import pickle
from collections import defaultdict
from playground.path_table import get_path_from_table
from playground.chair.chair import CHAIR
from playground._utils._colors import *
from pathlib import Path
import itertools
import os

try:
    # import spacy
    from nltk.stem import WordNetLemmatizer
    import nltk
    from nltk.corpus import wordnet
except ModuleNotFoundError as e:
    print_error(
        f"""CHAIR / AMBER parser uses library `spacy` and `nltk`. You can run the following command to install:
pip install spacy nltk 'numpy<2'
python -m spacy download en_core_web_lg
"""
    )
    exit(1)

from typing import List, Dict, Set, Tuple, ClassVar, Optional


class BaseParser(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def PARSER_WORDS(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def SAFE_WORDS(self) -> Dict[str, Set[str]]:
        ...

    @abstractmethod
    def extract_nouns(
        self, caption: str
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Extracts object words from the given caption and returns four lists.

        Returns:
        - A list of extracted words in their original form.
        - A list of extracted words after lemmatization.
        - A list of starting character indices of the extracted words in the original caption.
        - A list of ending character indices of the extracted words in the original caption.

        The start and end indices follow a half-open interval [start, end).
        """
        ...


class AmberParser(BaseParser):
    # This part of code is adapted from https://github.com/junyangwang0410/AMBER

    name = "amber"

    def __init__(self):
        super().__init__()

        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        _association = json.load(
            open(
                os.path.join(get_path_from_table("AMBER path"), "relation.json"),
                "r",
                encoding="utf-8",
            )
        )
        _hallucination_words: List[str] = []
        _safe_words = defaultdict(set)

        for word1 in _association.keys():
            _hallucination_words.append(word1)
            for word2 in _association[word1]:
                _hallucination_words.append(word2)
                _safe_words[word1].add(word2)
                _safe_words[word2].add(word1)

        _global_safe_words = []
        with open(
            os.path.join(get_path_from_table("AMBER path"), "safe_words.txt"),
            "r",
            encoding="utf-8",
        ) as safe_file:
            for line in safe_file:
                line = line.split("\n")[0]
                _global_safe_words.append(line)

        # Ensuring the reproducibility
        _hallucination_words = sorted(set(_hallucination_words))

        for safe_word in _global_safe_words:
            if safe_word in _hallucination_words:
                _hallucination_words.remove(safe_word)

        AMBER_WORDS: List[str] = _hallucination_words
        SAFE_WORDS: defaultdict[str, Set[str]] = _safe_words

        self.lemmatizer = WordNetLemmatizer()

        self._PARSER_WORDS = AMBER_WORDS
        self._SAFE_WORDS = SAFE_WORDS

    @property
    def PARSER_WORDS(self):
        return self._PARSER_WORDS

    @property
    def SAFE_WORDS(self):
        return self._SAFE_WORDS

    def extract_nouns(
        self, caption: str
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        # 1. Tokenize the caption
        tokens = nltk.word_tokenize(caption)
        tagged = nltk.pos_tag(tokens)

        # 2. Get the start and end position of each amber token
        start_poses = []
        end_poses = []
        end_pos = 0
        for nltk_token in tokens:
            start_pos = caption.find(nltk_token, end_pos)
            if start_pos == -1:
                start_poses.append(None)
                end_poses.append(None)
            else:
                end_pos = start_pos + len(nltk_token)
                start_poses.append(start_pos)
                end_poses.append(end_pos)

        # 3. Extract amber nouns
        tokens_filtered = []
        lemmatized_tokens_filtered = []
        start_poses_filtered = []
        end_poses_filtered = []

        for t, (w, p), s, e in zip(tokens, tagged, start_poses, end_poses):
            if p.startswith("NN"):
                l = self.lemmatizer.lemmatize(w)
                if l in self.PARSER_WORDS and s is not None and e is not None:
                    tokens_filtered.append(t)
                    lemmatized_tokens_filtered.append(l)
                    start_poses_filtered.append(s)
                    end_poses_filtered.append(e)

        return (
            tokens_filtered,
            lemmatized_tokens_filtered,
            start_poses_filtered,
            end_poses_filtered,
        )


class ChairParser(BaseParser):
    # This part of code is adapted from https://github.com/Maxlinn/CHAIR-metric-standalone/blob/main/chair.py

    name = "chair"

    synonyms_txt = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
    """.strip()

    def __init__(self):
        super().__init__()

        CHAIR_CACHE = Path("./playground/chair/chair.pkl")
        self.evaluator: CHAIR = pickle.load(open(CHAIR_CACHE, "rb"))
        print_note(f"Loaded evaluator from cache: {CHAIR_CACHE}")

        self._synonyms = defaultdict(set)
        self._PARSER_WORDS = []

        # read in synonyms
        synonyms = self.synonyms_txt.splitlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        for line in synonyms:
            self._PARSER_WORDS.append(line[0])
            for s1, s2 in itertools.permutations(line, 2):
                self._synonyms[s1].add(s2)

        self._SAFE_WORDS = self._synonyms

    @staticmethod
    def get_token_indices(caption: str, tokens: list[str]):
        start_poses: list[Optional[int]] = []
        end_poses: list[Optional[int]] = []
        end_pos = 0
        for token in tokens:
            start_pos = caption.find(token, end_pos)
            if start_pos == -1:
                start_poses.append(None)
                end_poses.append(None)
            else:
                end_pos = start_pos + len(token)
                start_poses.append(start_pos)
                end_poses.append(end_pos)

        return start_poses, end_poses

    def extract_nouns(self, caption: str):
        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.evaluator.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # words = [singularize(w) for w in words]
        origin_words = words
        words = lemmas_sent

        # replace double words
        i = 0
        double_words = []
        origin_double_words = []  # ===== HalTrapper: Added
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            origin_double_word = " ".join(
                origin_words[i : i + 2]
            )  # ===== HalTrapper: Added
            if double_word in self.evaluator.double_word_dict:
                double_words.append(self.evaluator.double_word_dict[double_word])
                origin_double_words.append(
                    origin_double_word
                )  # ===== HalTrapper: Added
                i += 2
            else:
                double_words.append(words[i])
                origin_double_words.append(origin_words[i])  # ===== HalTrapper: Added
                i += 1
        words = double_words
        double_words = origin_double_words  # ===== HalTrapper: Added

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # get synonyms for all words in the caption
        # TODO: check what if then?
        idxs = [
            idx
            for idx, word in enumerate(words)
            if word in set(self.evaluator.mscoco_objects)
        ]
        # idxs = [idxs[idx] for idx, word in enumerate(words) \
        #         if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.evaluator.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.evaluator.inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption

        start_poses, end_poses = self.get_token_indices(caption.lower(), double_words)

        start_poses = [start_poses[i] for i in idxs]
        end_poses = [end_poses[i] for i in idxs]

        # Filter out entries with None in start_poses or end_poses
        filtered = [
            (w, nw, sp, ep)
            for (w, nw, sp, ep) in zip(words, node_words, start_poses, end_poses)
            if sp is not None and ep is not None
        ]

        # Unzip filtered tuples back into separate lists
        if filtered:
            (words, node_words, start_poses_filtered, end_poses_filtered) = zip(
                *filtered
            )
            words = list(words)
            node_words = list(node_words)
            start_poses_filtered = list(start_poses_filtered)
            end_poses_filtered = list(end_poses_filtered)
        else:
            # No valid entries left
            words, node_words, start_poses_filtered, end_poses_filtered = [], [], [], []

        return words, node_words, start_poses_filtered, end_poses_filtered

    @property
    def PARSER_WORDS(self):
        return self._PARSER_WORDS

    @property
    def SAFE_WORDS(self):
        return self._SAFE_WORDS

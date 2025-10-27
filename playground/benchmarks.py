import re
import os
import random
import json
from pathlib import Path
from torch.utils.data import Dataset
from .path_table import get_path_from_table
from ._utils._path import safe_open
from ._utils._colors import *

from abc import ABC, abstractmethod

from typing import Optional, Type, Tuple, Dict, Any, ClassVar
from ._utils._path import PathObj, load_structured_file, save_structured_file


class BenchBase(Dataset[tuple[str, Optional[PathObj], Optional[dict[Any, Any]]]], ABC):
    name: ClassVar[str]
    registry: ClassVar[dict[str, Type["BenchBase"]]] = {}

    def __init_subclass__(cls, cmd_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cmd_name is None:
            cmd_name = cls.__name__

        cls.registry[cmd_name.lower()] = cls

    @abstractmethod
    def get_score(self, log_list: list[Any], log_file_path: PathObj) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, PathObj, Optional[Dict[Any, Any]]]:
        ...

    # deprecated alias
    def eval(self, *args, **kwargs):
        self.get_score(*args, **kwargs)


class CHAIR(BenchBase):
    name = "chair"
    fixed_500 = [
        108193,
        538005,
        187235,
        101535,
        533743,
        277383,
        91545,
        194434,
        263612,
        18022,
        456969,
        262189,
        219792,
        200938,
        39718,
        137003,
        554156,
        50778,
        492254,
        522791,
        547013,
        170852,
        220685,
        240082,
        72843,
        471789,
        34922,
        332908,
        108853,
        106331,
        240742,
        484175,
        62661,
        257270,
        513946,
        295628,
        299716,
        333737,
        486233,
        322143,
        512151,
        187348,
        106411,
        443426,
        503585,
        413278,
        128392,
        258036,
        134213,
        43947,
        238404,
        267321,
        568090,
        207844,
        135266,
        574057,
        212663,
        158798,
        374270,
        483447,
        510383,
        320461,
        169972,
        251098,
        65889,
        292206,
        9548,
        46345,
        344498,
        147681,
        198247,
        427956,
        452721,
        494345,
        468018,
        9450,
        162130,
        415619,
        288874,
        40317,
        212122,
        564830,
        58933,
        270571,
        334782,
        412355,
        475917,
        103509,
        391915,
        563015,
        396692,
        417632,
        562737,
        53145,
        379605,
        409646,
        279784,
        184606,
        391648,
        286055,
        378873,
        340843,
        179930,
        132587,
        281977,
        452084,
        433652,
        483656,
        348059,
        174258,
        27009,
        543034,
        290558,
        126107,
        237350,
        14248,
        271148,
        562885,
        28826,
        94513,
        4742,
        543006,
        469046,
        279491,
        538775,
        130948,
        503210,
        328785,
        438737,
        399744,
        133125,
        490244,
        145032,
        103499,
        555009,
        223955,
        553482,
        453485,
        306949,
        310711,
        459195,
        140513,
        404128,
        209431,
        494393,
        67065,
        571449,
        498392,
        428847,
        92801,
        393811,
        67502,
        391320,
        487222,
        136533,
        423104,
        271986,
        474609,
        71877,
        535135,
        381173,
        94619,
        31202,
        87204,
        225658,
        172995,
        398158,
        197193,
        323983,
        125100,
        340701,
        213255,
        77628,
        533506,
        488869,
        274063,
        551494,
        179765,
        469085,
        256301,
        246233,
        421882,
        345361,
        387216,
        182441,
        369345,
        551896,
        274017,
        27755,
        344410,
        118406,
        458702,
        457817,
        161567,
        132689,
        286478,
        291712,
        351053,
        394715,
        31711,
        446113,
        185473,
        352495,
        486009,
        240323,
        435384,
        154155,
        20913,
        129579,
        204853,
        364125,
        470885,
        342260,
        543696,
        215245,
        558673,
        251330,
        37670,
        466789,
        370209,
        97790,
        247378,
        144562,
        113571,
        203061,
        309889,
        45197,
        271259,
        379101,
        367313,
        34701,
        463174,
        74733,
        83761,
        74963,
        542077,
        227326,
        144863,
        543218,
        2529,
        483001,
        196521,
        173984,
        494578,
        404618,
        392564,
        329088,
        568333,
        19087,
        562843,
        280612,
        442536,
        67075,
        481446,
        537369,
        269344,
        203639,
        122589,
        150365,
        46405,
        503005,
        193388,
        275449,
        561856,
        427245,
        107216,
        125228,
        337188,
        543364,
        533750,
        89395,
        230881,
        210980,
        292581,
        546095,
        76873,
        50521,
        98350,
        231945,
        544299,
        329653,
        381247,
        416862,
        471023,
        251042,
        422893,
        272738,
        262016,
        422040,
        252127,
        437789,
        573209,
        402433,
        23623,
        63882,
        178578,
        263223,
        549136,
        178835,
        414216,
        436349,
        435909,
        252280,
        64332,
        54088,
        244184,
        210522,
        361551,
        303499,
        558955,
        114033,
        326308,
        472375,
        294475,
        208107,
        111909,
        552352,
        503399,
        273642,
        122413,
        577887,
        538888,
        177913,
        129735,
        300753,
        98392,
        19559,
        254237,
        309514,
        334371,
        262391,
        7355,
        443492,
        74000,
        567494,
        14271,
        71938,
        246532,
        159223,
        214646,
        32887,
        493682,
        565582,
        549338,
        79836,
        448705,
        125404,
        101223,
        335631,
        12179,
        572734,
        308115,
        350447,
        57244,
        517636,
        511929,
        248085,
        337826,
        424799,
        198518,
        564123,
        13943,
        162627,
        260695,
        281220,
        306186,
        508339,
        261535,
        320796,
        159684,
        205002,
        340155,
        287366,
        440895,
        187624,
        213650,
        441873,
        567812,
        15267,
        62298,
        337055,
        126030,
        61233,
        428366,
        95561,
        41247,
        45536,
        391689,
        80357,
        151130,
        23489,
        92656,
        218855,
        387990,
        503200,
        11449,
        563311,
        460312,
        278934,
        64116,
        63480,
        494182,
        547431,
        282553,
        352194,
        455741,
        410337,
        255401,
        260382,
        164885,
        149272,
        67431,
        113440,
        191613,
        574928,
        150685,
        542910,
        418275,
        437732,
        485909,
        287714,
        185512,
        90804,
        284143,
        407532,
        83408,
        122317,
        309678,
        271900,
        491000,
        543231,
        359184,
        32510,
        44825,
        463610,
        83547,
        499181,
        111109,
        12010,
        403104,
        525702,
        489739,
        274494,
        208825,
        336873,
        300972,
        35368,
        249482,
        420658,
        331753,
        288762,
        33216,
        537802,
        210990,
        339120,
        80066,
        540564,
        77178,
        175205,
        351808,
        176592,
        444152,
        449302,
        110762,
        118134,
        102353,
        329379,
        216437,
        245750,
        11721,
        168434,
        360510,
        291696,
        303867,
        87399,
        37038,
        406611,
        285,
        299734,
        60347,
        125435,
        532901,
        257920,
        89643,
        248284,
        501760,
        160393,
        11034,
        273715,
        281259,
        261180,
        214961,
        303814,
        121196,
        352228,
        316842,
        56633,
        24247,
        282231,
        183735,
    ]

    def __init__(self, total_sampled_images: int = 500, fixed: bool = True) -> None:
        super().__init__()
        self.COCO_PATH = get_path_from_table("COCO path")

        self.coco_dirpath = self.COCO_PATH
        self.sample_n = total_sampled_images
        img_files = os.listdir(self.coco_dirpath)
        if fixed:
            self.img_files = [
                os.path.join(self.coco_dirpath, f"COCO_val2014_{coco_id:012}.jpg")
                for coco_id in self.fixed_500[:total_sampled_images]
            ]
        else:
            random.shuffle(img_files)
            self.img_files = img_files[: self.sample_n]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        return (
            "Please help me describe the image in detail.",
            os.path.join(self.coco_dirpath, img_file),
            {"COCO_id": img_id},
        )

    def get_score(self, log_list, log_file_path) -> None:
        chair_file_path = os.path.splitext(log_file_path)[0] + "--chair-style.jsonl"
        save_path = os.path.splitext(log_file_path)[0] + "--chair-output.json"

        with safe_open(chair_file_path, "w") as f:
            for cur_dict in log_list:
                img_id = cur_dict["COCO_id"]
                output_text = cur_dict["response"]
                json.dump({"image_id": img_id, "caption": output_text}, f)
                f.write("\n")

        this_dir_path = Path(__file__).resolve().parent
        chairpy_dir_path = this_dir_path / "chair"

        command = f"""python "{chairpy_dir_path/'chair.py'}" \\
    --cap_file "{chair_file_path}" \\
    --cache "{chairpy_dir_path/'chair.pkl'}" \\
    --save_path "{save_path}"
"""

        if os.system(command):
            print_error(
                f"Failed to calculate the CHAIR score. You can run the following command to manually perform the task:\n{command}"
            )


class AMBER(BenchBase):
    name = "amber"
    SPLITS = (
        ("a", "all", "query_all.json"),
        ("g", "generative", "query_generative.json"),
        ("d", "discriminative", "query_discriminative.json"),
        ("de", "discriminative-existence", "query_discriminative-existence.json"),
        ("da", "discriminative-attribute", "query_discriminative-attribute.json"),
        ("dr", "discriminative-relation", "query_discriminative-relation.json"),
    )
    NEGS = {
        "no",
        "not",
        "none",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "nobody",
        "without",
        "zero",
        "0",
    }
    RE_NEG_PATTERN = r"\b(?:" + "|".join(re.escape(word) for word in NEGS) + r")\b"

    def __init__(self, split: str = "a", change_prompt: bool = False) -> None:
        super().__init__()
        self.change_prompt = change_prompt
        self.AMBER_ROOT = get_path_from_table("AMBER path")

        split = split.lower()

        all_splits = []
        for split_sym, split_name, query_file_name in self.SPLITS:
            if split in ((split_sym, split_name)):
                self.split = split_sym
                self.query_file_name = query_file_name
                break
            else:
                all_splits.append(split_sym)
                all_splits.append(split_name)
        else:
            raise ValueError(
                f"AMBER: `split` should be in {all_splits}, got {repr(split)}."
            )

        self.query_file_path = os.path.join(
            self.AMBER_ROOT, "query", self.query_file_name
        )

        self.amber_questions = load_structured_file(self.query_file_path)

        self.image_dirpath = os.path.join(self.AMBER_ROOT, "image")

    def __len__(self):
        return len(self.amber_questions)

    def __getitem__(self, index):
        item = self.amber_questions[index]
        image = item["image"]
        text = item["query"]
        amber_id = item["id"]

        if self.change_prompt:
            if amber_id >= 1005:
                prompt = f"{text} Please answer this question with one word."
            else:
                prompt = "Please help me describe the image in detail."
        else:
            prompt = text

        return (
            prompt,
            os.path.join(self.image_dirpath, image),
            {
                "AMBER_id": amber_id,
            },
        )

    def get_score(self, log_list, log_file_path) -> None:
        amber_file_path = os.path.splitext(log_file_path)[0] + "--amber-style.json"

        output_list = []
        for cur_dict in log_list:
            amber_id = cur_dict["AMBER_id"]
            if 1 <= amber_id <= 1004:
                output_list.append({"id": amber_id, "response": cur_dict["response"]})
            elif amber_id >= 1005:
                output_list.append(
                    {
                        "id": amber_id,
                        "response": (
                            "No"
                            if bool(
                                re.search(
                                    self.RE_NEG_PATTERN,
                                    cur_dict["response"].lower(),
                                    flags=re.IGNORECASE,
                                )
                            )
                            else "Yes"
                        ),
                    }
                )
            else:
                raise

        save_structured_file(output_list, amber_file_path, "w")

        this_dir_path = Path(__file__).resolve().parent
        amber_inference_path = this_dir_path / "amber"

        command = f"""python "{amber_inference_path/'inference.py'}" \\
    --word_association "{os.path.join(self.AMBER_ROOT, 'relation.json')}" \\
    --safe_words "{os.path.join(self.AMBER_ROOT, 'safe_words.txt')}" \\
    --inference_data "{amber_file_path}" \\
    --annotation "{os.path.join(self.AMBER_ROOT, 'annotations.json')}" \\
    --metrics "{os.path.join(self.AMBER_ROOT, 'metrics.txt')}" \\
    --evaluation_type "{self.split}"
"""

        if os.system(command):
            print_error(
                f"Failed to calculate the AMBER score. You can run the following command to manually perform the task:\n{command}"
            )
            print_note(
                f"""AMBER uses library `spacy` and `nltk`. You can run the following command to install:
pip install spacy nltk 'numpy<2'
python -m spacy download en_core_web_lg
"""
            )

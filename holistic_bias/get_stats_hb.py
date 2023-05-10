import functools
import re
import typing as tp
from collections import Counter
from pathlib import Path
import sys
sys.path.append('/private/home/benjaminmuller/dev/biases/ResponsibleNLP')
print(sys.path)
from holistic_bias.src.sentences import HolisticBiasSentenceGenerator

#from stopes.modules.preprocess.multiproc_line_processor import (
#    MultiprocLineProcessorCallback,
#)


def _parse_filename(filename: str) -> tp.Tuple[str, str]:
    (_, langs, _) = Path(filename).name.split(".", maxsplit=2)
    sp = langs.split("-")
    return (sp[0], sp[1])


class CountHolisticBiasCallback(object):
    """
    This is the core of the counting. It does the counting based on Holistic_Bias
    list of nouns and noun phrases, return counters, and merge them as needed.
    """

    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
    ) -> None:
        #super().__init__(
        #    outfile_prefix=outfile_prefix,
        #    input_file=input_file,
        ##    input_file_idx=input_file_idx,
        #    output_dir=output_dir,
        #    offset_start=offset_start,
        #    offset_end=offset_end,
        #    merging=merging,
        #)

        (left_lang, right_lang) = _parse_filename(input_file)

        self.eng_is_left = left_lang == "eng"
        self.file_lang = right_lang if self.eng_is_left else left_lang

        #if not merging:
        #    # we don't need any of these when we merge, so save on the init cost
        output_dir = Path(output_dir)
        dataset_version = 'v1.1'
        holistic_bias_data = HolisticBiasSentenceGenerator(output_dir/outfile_prefix, dataset_version=dataset_version) 
        self.noun_phrases = holistic_bias_data.get_compiled_noun_phrases(dataset_version=dataset_version)
        # we want to match noun phrases disregarding the undefinite article
        # that's added by default, hack around it (ideally this is an option in HB)
        reg = re.compile(r"^an? ", re.IGNORECASE)
        self.noun_phrases["noun_phrase_simple"] = self.noun_phrases[
            "noun_phrase"
        ].apply(lambda s: reg.sub("", s))
        self.noun_phrases["noun_phrase_re"] = self.noun_phrases.apply(
            lambda r: re.compile(
                f"\\b({r['noun_phrase_simple']}|{r['plural_noun_phrase']})\\b",
                re.IGNORECASE,
            ),
            axis=1,
        )

        # build a regexp to find gendered noun mentions
        # one regexp per group_gender:
        # 'female' => r"..."
        # 'male' => r"..."
        self.gender_regs = {}
        NOUNS = holistic_bias_data.get_nouns(dataset_version)
        for group_gender, gender_noun_tuples in NOUNS.items():
            r_string = "\\b("
            for noun, plural_noun in gender_noun_tuples:
                r_string += f"{re.escape(noun)}|{re.escape(plural_noun)}|"
            r_string = r_string[:-1]
            r_string += ")\\b"
            self.gender_regs[group_gender] = re.compile(r_string, re.IGNORECASE)

        # Counters
        self.count_np: Counter = Counter()
        self.count_gender: Counter = Counter()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        for _idx, line in lines_with_number:
            # depends on the input file if eng is on the left col or the right col
            (_s, lang1, lang2) = line.split("\t", maxsplit=3)
            eng = lang1 if self.eng_is_left else lang2

            self.count_np["_total"] += 1
            # for gender, we count words instead of lines, so we do basic tokenization (eng only)
            self.count_gender["_total"] += len(eng.split())
            for _idx, w in self.noun_phrases.iterrows():
                if w["noun_phrase_re"].search(eng):
                    key = (
                        f"{w['bucket']}\t"
                        f"{w['axis']}\t"
                        f"{w['noun_gender']}\t"
                        f"{w['descriptor_preference']}"
                    )
                    self.count_np[key] += 1
            for group_gender, reg in self.gender_regs.items():
                self.count_gender[group_gender] += len(reg.findall(eng))
    
    def process_single_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        # iterate over lines
        for line in lines_with_number:
            # depends on the input file if eng is on the left col or the right col
            sentence = line.strip()
            #(_s, sentence) = line.split("\t", maxsplit=3)
            # lines counter
            self.count_np["_total"] += 1
            # for gender, we count words instead of lines, so we do basic tokenization (eng only)
            self.count_gender["_total"] += len(sentence.split()) # count words
            for _idx, w in self.noun_phrases.iterrows():
                # for each noun_phrases: e.g. (working class man |) (e.g match both plural/singular: bro who is an amputee|bros who are amputees)
                # ==> count if the phrase includes it: if it does: append counter
                if w["noun_phrase_re"].search(sentence):
                    key = (
                        f"{w['bucket']}\t"
                        f"{w['axis']}\t"
                        f"{w['noun_gender']}\t"
                        f"{w['descriptor_preference']}"
                    )
                    # count the occurence for the category of what was matched (bucket, gender, ) : bucket: type of descriptor; value in the bucket ; 
                    self.count_np[key] += 1
                    
            for group_gender, reg in self.gender_regs.items():
                
                # match to list of nouns that are feminine and masculine 
                self.count_gender[group_gender] += len(reg.findall(sentence))


    def final_result(self) -> tp.Tuple[str, Counter, Counter]:
        return (self.file_lang, self.count_gender, self.count_np)

if __name__ == '__main__':
    #from stopes.eval.holistic_bias.callback import CountHolisticBiasCallback

    hb_counter = CountHolisticBiasCallback(outfile_prefix='tmp2', input_file='./test.en-en.read.txt', input_file_idx=0,
                                        output_dir='test2', offset_start=None, offset_end=None, merging=False)
    lines = []
    with open('/private/home/benjaminmuller/dev/biases/data/eval_test/test.txt', 'r') as f:
        hb_counter.process_single_lines(f)

    print(hb_counter.final_result)

    import pdb
    pdb.set_trace()
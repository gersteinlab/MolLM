import json
import os.path
import time
import traceback
import urllib
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import backoff
import nltk
from transformers import BertTokenizer

from db import DB
from task import Task
from lock import *


class TextTask(Task[Tuple[int, List[int]]]):
    def get_batch(self, batch_size: int) -> List[Tuple[int, List[int]]]:
        try:
            select_sql = """SELECT offset
                                FROM v2_mol_batches 
                                WHERE in_progress = 0
                                LIMIT %s"""

            results = DB.fetch_all(select_sql, (batch_size,), commit=False)

            if results is None:
                return []

            random_id_offsets = [row['offset'] for row in results]

            # Find real CIDs
            cid_lists = []
            for random_id_offset in random_id_offsets:
                select_sql = """SELECT cid 
                                    FROM v2_mols
                                    ORDER BY cid
                                    LIMIT 1000
                                    OFFSET %s"""
                cids = DB.fetch_all(select_sql, (random_id_offset,), commit=True)
                cids = [row['cid'] for row in cids]
                cid_lists.append((random_id_offset, cids))

            # Mark as in_process
            update_sql = """UPDATE v2_mol_batches
                                SET in_progress = 1
                                WHERE offset = %s"""
            for random_id_offset in random_id_offsets:
                DB.execute(update_sql, (random_id_offset,), commit=False)
            DB.commit()

        except Exception as e:
            print("An error occurred: ", e)
            print(traceback.format_exc())
            DB.rollback()
            return []

        return cid_lists

    def process(self, batch):
        random_id_offset, cids = batch
        tokenizer = BertTokenizer.from_pretrained('../bert_pretrained/')

        success = 0
        success_tuples = []

        wait_time = 15
        max_wait_time = 600
        for idx, cid in enumerate(cids):
            try:
                smiles, all_text, should_slow_down = get_smiles_and_text(cid, tokenizer)

                # Throttle
                if should_slow_down:
                    print(f"Throttling control red, waiting for {wait_time} seconds")
                    time.sleep(wait_time)
                    wait_time = min(2 * wait_time, max_wait_time)
                else:
                    wait_time = 15  # Reset the wait time

                success_tuples.append((cid, smiles, all_text))
                success += 1
            except Exception as e:
                # print(traceback.format_exc())
                # print(e)
                pass
            if idx % 100 == 0:
                print(f'Progress: {idx + 1} / 1000 (success {success})')

        # Save
        indices_to_results = defaultdict(list)
        for cid, smiles, all_text in success_tuples:
            indices_to_results[get_path_indices(cid)].append((cid, smiles, all_text))

        for indices, success_subset in indices_to_results.items():
            folder, zip_path = get_folder_and_zip_path(indices)
            Path(folder).mkdir(parents=True, exist_ok=True)
            try:
                acquire_lock(zip_path)
                with ZipFile(zip_path, 'a') as zip_file:
                    existing_files = zip_file.namelist()
                    for cid, smiles, all_text in success_subset:
                        # SMILES
                        smiles_file_name = f'mol_{cid}_smiles.txt'
                        if smiles_file_name not in existing_files:
                            zip_file.writestr(smiles_file_name, smiles)
                        # Text
                        text_file_name = f'mol_{cid}_text.txt'
                        if text_file_name not in existing_files:
                            zip_file.writestr(text_file_name, '\n'.join(all_text))
            finally:
                release_lock(zip_path)

        # Mark chunk as done
        update_sql = """UPDATE v2_mol_batches
                            SET finished=1, success=%s
                            WHERE offset = %s"""
        DB.execute(update_sql, (success, random_id_offset))


def find_section_with_heading(sections, heading):
    for section in sections:
        if section['TOCHeading'] == heading:
            return section
    return None


def string_w_markup_to_text(string_w_markup) -> str:
    strings = []
    for string in string_w_markup:
        strings.append(string['String'])
    return ' '.join(strings)


def ends_in_special_char(s: str) -> bool:
    return s and not s[-1].isalpha()


def process_strings(strings) -> List[str]:
    strings = [s if ends_in_special_char(s) else s + '.' for s in strings]
    strings = [s.replace('**', '') for s in strings]
    strings = [s.replace('\n', '') for s in strings]
    strings = [s.replace('\r', '') for s in strings]
    return strings


def information_to_text(information_section, clean_strings=True) -> List[str]:
    strings = []
    for info in information_section:
        if 'StringWithMarkup' not in info['Value']:
            continue
        strings.append(string_w_markup_to_text(info['Value']['StringWithMarkup']))
    if clean_strings:
        strings = process_strings(strings)
    return strings


def extract_description(sections):
    section = find_section_with_heading(sections, "Names and Identifiers")
    description = find_section_with_heading(section['Section'], 'Record Description')
    description = ' '.join(information_to_text(description['Information']))
    if len(description) < 150 and 'with data available' in description:
        raise ValueError('With data available description is useless')
    return description


def extract_properties(sections):
    section = find_section_with_heading(sections, 'Chemical and Physical Properties')
    experimental = find_section_with_heading(section['Section'], 'Experimental Properties')

    property_strings = []
    for property_section in experimental['Section']:
        heading = property_section['TOCHeading'] + ': '
        property_strings.append(heading + ' '.join(information_to_text(property_section['Information'])))

    return ' '.join(property_strings)


def extract_headings(sections, section_name, headers):
    if isinstance(section_name, str):
        section = find_section_with_heading(sections, section_name)
    else:
        section = section_name

    strings = []
    for subsection in section['Section']:
        if subsection['TOCHeading'] in headers:
            heading = subsection['TOCHeading'] + ': '
            strings.append(heading + ' '.join(information_to_text(subsection['Information'])))

    return ' '.join(strings)


def extract_drug_info(sections):
    return extract_headings(sections, 'Drug and Medication Information',
                            ['Drug Indication', 'LiverTox Summary', 'Drug Classes'])


def extract_pharmacology_info(sections):
    return extract_headings(sections, 'Pharmacology and Biochemistry',
                            ['Pharmacodynamics', 'MeSH Pharmacological Classification'])


def extract_toxicity_info(sections):
    section = find_section_with_heading(sections, 'Toxicity')
    section = find_section_with_heading(section['Section'], 'Toxicological Information')
    return extract_headings(sections, section, ['Toxicity Summary'])


def split_text(text: str, tokenizer: BertTokenizer, max_length: int = 250, initial_guess: int = 10, step: int = 2) -> \
        List[str]:
    # Sentence tokenization
    sentences = nltk.sent_tokenize(text)

    result = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Estimate the number of tokens
        tokens = tokenizer.tokenize(sentence)
        token_length = len(tokens)

        # If adding this sentence does not exceed the max length
        if current_length + token_length <= max_length:
            current_chunk.append(sentence)
            current_length += token_length
        else:
            # If the current chunk is empty, we have a sentence that exceeds the max_length, we have to split it
            if not current_chunk:
                for i in range(0, len(tokens), max_length):
                    chunk = tokens[i:i + max_length]
                    result.append(tokenizer.convert_tokens_to_string(chunk))
                continue
            # Otherwise, add the current chunk to the result and start a new chunk
            result.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = token_length

    # Add the last chunk
    if current_chunk:
        result.append(' '.join(current_chunk))

    return result


def backoff_hdlr(details):
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))


@backoff.on_exception(backoff.expo, urllib.error.HTTPError, max_tries=5, on_backoff=backoff_hdlr)
def get_from_url(cid):
    with urllib.request.urlopen(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON') as url:
        data = json.load(url)
        # Save to file
        folder_path, zip_path = get_folder_and_zip_path(get_path_indices(cid))
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        try:
            acquire_lock(zip_path)
            with ZipFile(zip_path, 'a') as zip_file:
                existing_files = zip_file.namelist()
                file_name = f'{cid}.json'
                if file_name not in existing_files:
                    zip_file.writestr(file_name, json.dumps(data))
        finally:
            release_lock(zip_path)

        ctrl = url.getheader('X-Throttling-Control')
        should_slow_down = ('Request Count status: Red' in ctrl or 'Request Time status: Red' in ctrl)
        return data, should_slow_down


def get_smiles_and_text(cid, tokenizer):
    data, should_slow_down = get_from_url(cid)

    json_sections = data['Record']['Section']

    # Find SMILES
    names = find_section_with_heading(json_sections, 'Names and Identifiers')
    descriptors = find_section_with_heading(names['Section'], 'Computed Descriptors')
    smiles_section = find_section_with_heading(descriptors['Section'], 'Canonical SMILES')
    smiles = smiles_section['Information'][0]['Value']['StringWithMarkup'][0]['String']

    # Find text
    extractors = [
        (extract_description, True),
        (extract_properties, False),
        (extract_drug_info, False),
        (extract_pharmacology_info, False),
        (extract_toxicity_info, False)
    ]

    all_text = []
    for extractor, required in extractors:
        try:
            all_text.append(extractor(json_sections))
        except Exception as e:
            if required:
                raise ValueError(f"Required extractor failed: {e}")

    all_text_split = []
    for text in all_text:
        all_text_split.extend(split_text(text, tokenizer))

    return smiles, all_text_split, should_slow_down


def get_path_indices(num):
    # Break down the input number into two separate numbers
    num_str = str(num)
    while len(num_str) < 9:
        num_str = "0" + num_str
    part1 = num_str[:3]
    part2 = num_str[3:6]

    # Convert each part into a string and pad with leading zeroes
    part1 = part1.zfill(3)
    part2 = part2.zfill(3)
    return part1, part2


def get_folder_and_zip_path(indices):
    part1, part2 = indices
    folder = f"output-text/{part1}/"
    tar = folder + f"{part2}.zip"
    return folder, tar

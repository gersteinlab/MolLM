import traceback
import xml.etree.cElementTree as ET
from typing import List, Tuple

from db import DB
from task import Task


class CandidateTask(Task[Tuple[int, int]]):
    def get_batch(self, batch_size: int) -> List[Tuple[int, int]]:
        try:
            # SQL query to select rows where in_process is 0 (or False),
            # limiting the result to the batch size provided.
            select_sql = """SELECT starting_id, ending_id
                                FROM xml_files 
                                WHERE in_process = 0
                                LIMIT %s"""

            # Use the DB class method to fetch all results from the database.
            results = DB.fetch_all(select_sql, (batch_size,), commit=False)

            # If no results are returned, return an empty list.
            if results is None:
                return []

            # Create a list of tuples from the results.
            id_pairs = [(row['starting_id'], row['ending_id']) for row in results]

            # SQL query to update in_process to 1 for the rows just fetched.
            update_sql = """UPDATE xml_files 
                                SET in_process = 1 
                                WHERE starting_id = %s AND ending_id = %s"""

            # Execute the update query for each pair of IDs.
            for pair in id_pairs:
                DB.execute(update_sql, pair, commit=False)

            # Commit the transaction
            DB.commit()

        except Exception as e:
            print("An error occurred: ", e)
            print(traceback.format_exc())
            DB.rollback()
            return []

        return id_pairs

    def process(self, ids: Tuple[int, int]):
        start_id, end_id = ids
        file_name = f'../pubchem-xml/Compound_{start_id:09}_{end_id:09}.xml'
        compound_id = None
        atom_count = None
        atom_types = None

        about_to_read_smiles = False
        smiles = None

        processed = 0
        context = ET.iterparse(file_name, events=("end",))
        for event, elem in context:
            if event != 'end':
                continue

            tag = elem.tag.removeprefix('{http://www.ncbi.nlm.nih.gov}')

            if tag == 'PC-CompoundType_id_cid':
                compound_id = int(elem.text)
                atom_count = 0
                atom_types = set()
                about_to_read_smiles = False
                smiles = None
            elif tag == 'PC-Element':
                atom_count += 1
                atom_types.add(elem.attrib['value'].upper())
            elif tag == 'PC-Urn_label' and elem.text == 'SMILES':
                about_to_read_smiles = True
            elif about_to_read_smiles and tag == 'PC-InfoData_value_sval':
                about_to_read_smiles = False
                smiles = elem.text
            elif tag == 'PC-Compound':
                atom_types = ''.join(sorted(atom_types))
                valid = atom_count and atom_types and smiles and atom_count < 512
                if valid:
                    update_sql = """REPLACE INTO all_molecules(
                            cid, xml_checked, xml_candidate, atom_count, atom_types, smiles)
                            VALUES(%s, 1, 1, %s, %s, %s)"""
                    DB.execute(update_sql, (compound_id, atom_count, atom_types, smiles), commit=False)
                else:
                    update_sql = """REPLACE INTO all_molecules(cid, xml_checked, xml_candidate) VALUES (%s, 1, 0)"""
                    DB.execute(update_sql, (compound_id,), commit=False)

                processed += 1
                if processed % 500 == 0:
                    DB.commit()

                elem.clear()

        # Mark XML file as done
        update_sql = """UPDATE xml_files
                            SET finished=1
                            WHERE starting_id = %s"""

        DB.execute(update_sql, (start_id,))

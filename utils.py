import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from collections import defaultdict

import json

import pandas as pd



CLASS_MAP={
    "MAN_MADE_EVENT": "MANMADE_DISASTER",
    "NATURAL_EVENT": "NATURAL_DISASTER"

}

def transform_label(y):
    return pd.Series(y).str.split(".", expand=True)[0]


def transform_probs(probs, clf):
    clf_map = defaultdict(list)
    for c in clf.classes_:
        clf_map[c.split(".")[0]].append(c)
    df_probs = pd.DataFrame(probs, columns=clf.classes_)
    for c_parent, children in clf_map.items():
        df_probs[c_parent] = df_probs[children].sum(axis=1)
    return df_probs[[c for c in clf_map]].idxmax(axis=1)

def get_w_children(node, base_node=None, split_paras=False):
    idx = 0
    if base_node is None:
        base_node = ET.Element('P')
    for child in node:
        if child.tag == "W":
            label = base_node.tag
            if base_node.attrib.get("TYPE"):
                label = f"{label}.{base_node.attrib['TYPE']}"
            label = f"B-{label}" if idx == 0 else f"I-{label}"
            if base_node.tag == "P":
                label = "O"
            yield child.text.strip(), label
            idx+=1
        else:
            yield from get_w_children(child, base_node=child)
    if split_paras and node.tag == "P":
        yield "<P>", "P"
        
        
def get_w_children_test(node, base_node=None, split_paras=False):
    idx = 0
    for child in node:
        if child.tag == "P":
            yield from [(t.strip(), "O") for t in child.text.split("  ")]
            if split_paras:
                yield "<P>", "P"

        
def split_tag(tag):
    return tuple(tag.split("-", 1)) if tag != "O" else (tag, None)


def extract_entities(tags):
    tags = list(tags)
    curr_entity = []
    entities = []
    for i, tag in enumerate(tags + ["O"]):
        # Add dummy tag in end to ensure the last entity is added to entities
        boundary, label = split_tag(tag)
        if curr_entity:
            # Exit entity
            if boundary in {"B", "O"} or label != curr_entity[-1][1]:
                start = i - len(curr_entity)
                end = i
                entity_label = curr_entity[-1][1]
                entities.append((entity_label, start, end))
                curr_entity = []
            elif boundary == "I":
                curr_entity.append((boundary, label))
        if boundary == "B":
            # Enter or inside entity
            assert not curr_entity, f"Entity should be empty. Found: {curr_entity}"
            curr_entity.append((boundary, label))
    return entities


def get_entity_info(bio_labels, tokens, text=None, spans=None):
    entities_info = extract_entities(bio_labels)
    entities = []
    for label, start, end in entities_info:
        entity_phrase = None
        if text and spans:
            start_char_idx = spans[start][0]
            end_char_idx = spans[end-1][1]
        entity_phrase = " ".join(f" {t} " for t in tokens[start:end])
        entities.append(dict(
            tokens=tokens[start:end],
            label=label,
            start=start,
            end=end,
            entity_phrase=entity_phrase))
    return entities


def xml_to_conll(xml_path, test=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parse_fn = get_w_children_test if test else get_w_children
    seq = list(parse_fn(root))
    return seq

def xml_to_json(xml_path, test=False):
    seq = xml_to_conll(xml_path, test=test)
    docid = xml_path.stem
    tokens, tags = zip(*seq)
    
    labels = Counter([t[2:] for t in tags if t.startswith(("B-MAN_MADE_EVENT", "B-NATURAL_EVENT"))])
    return {
        "docid": docid,
        "tokens": tokens, 
        "tags": tags, 
        "labels": labels
    }



def get_all_json(folder, test=False):
    files = Path(folder).glob("./*.xml")
    all_json = []
    for xml_path in files:
        json_data = xml_to_json(xml_path, test=test)
        all_json.append(json_data)
    return all_json

def process_lang(lang):
    lang_folder = Path("./data/raw/") / lang
    out_folder = Path("./data/processed/") / lang
    out_folder.mkdir(exist_ok=True)
    for folder in ["Train", "Test"]:
        in_folder = lang_folder / folder
        out_file = out_folder / f"{folder.lower()}.json"
        print(f"Processing {in_folder} to {out_file}")
        train_data = get_all_json(in_folder, test=folder=="Test")
        df = pd.DataFrame(train_data).to_json(out_file, orient="records", lines=True)
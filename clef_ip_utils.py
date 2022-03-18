from pathlib import Path
import re
import xml.etree.ElementTree as ET


btext = ["b", "i", "o", "u", "sub", "sup", "smallcaps"]
ptext = ["b", "i", "o", "u", "sub", "sup", "smallcaps", "br", "dl", "ul", "ol", "sl"]
supported_children = {
    "abstract": ["abst-problem", "abst-solution", "heading", "p"],
    "abst-problem": ["p"],
    "abst-solution": ["p"],
    "heading": [*btext],
    "p": [*ptext],
    "claim": ["claim-text"],
    "claim-text": [*ptext, "claim-text"],
    "b":         [     "i", "o", "u", "sub", "sup", "smallcaps"],
    "i":         ["b",      "o", "u", "sub", "sup", "smallcaps"],
    "o":         ["b", "i",      "u", "sub", "sup", "smallcaps"],
    "u":         ["b", "i", "o",      "sub", "sup", "smallcaps"],
    "sub":       ["b", "i", "o", "u", "sub", "sup", "smallcaps", "sub2", "sup2"],
    "sup":       ["b", "i", "o", "u", "sub",        "smallcaps", "sub2", "sup2"],
    "smallcaps": ["b", "i", "o", "u", "sub", "sup",              "sub2", "sup2"],
    "sub2":      ["b", "i", "o", "u", "sub", "sup", "smallcaps",         "sup2"],
    "sup2":      ["b", "i", "o", "u", "sub", "sup", "smallcaps", "sub2"        ],
    "br": [],
    "dl": ["dt", "dd"],
    "ul": ["li"],
    "ol": ["li"],
    "sl": ["li"],
    "dt": [*btext],
    "dd": [*ptext],
    "li": [*ptext]
}
requires_whitespace = {
    "abstract": True,
    "abst-problem": True,
    "abst-solution": True,
    "heading": True,
    "p": True,    
    "claim": True,
    "claim-text": True,
    "b": False,
    "i": False,
    "o": False,
    "u": False,
    "sub": False,
    "sup": False,
    "smallcaps": False,
    "sub2": False,
    "sup2": False,
    "br": True,
    "dl": True,
    "ul": True,
    "ol": True,
    "sl": True,
    "dt": True,
    "dd": True,
    "li": True
}

def _get_text_from_tag(tag: ET.Element):
    final_text = tag.text or ""
    for child in tag:            
        if child.tag not in supported_children[tag.tag]:
            if child.tail is not None:
                final_text += " " + child.tail
            continue
        
        child_text = _get_text_from_tag(child)
        
        if requires_whitespace[child.tag] and final_text != "":
            final_text += " "
        
        final_text += child_text
        
        if child.tail is not None:
            if requires_whitespace[child.tag]:
                final_text += " "
            final_text += child.tail

    final_text = re.sub(r"\s+", " ", final_text)

    return final_text.strip()

def extract_content_from_patent(
    patent_file: Path,
    add_section_tags=False,
    abstract_only=False
):
    xml_document_root = ET.parse(patent_file)
    
    abstract = xml_document_root.find(".//abstract[@lang='EN']")
    abstract_text = _get_text_from_tag(abstract) if abstract is not None else ""
    if add_section_tags:
        abstract_text = f"[abstract] {abstract_text}"    

    if abstract_only:
        return abstract_text, []

    claims = xml_document_root.findall(".//claims[@lang='EN']/claim")   
    if patent_file.stem.startswith("EP"):
        # ^ EPO patent
        claims_texts = [_get_text_from_tag(claim) for claim in claims]
    else:
        # ^ WO patent
        assert len(claims) == 1  
        # ^ WO patents contain only one <claim> tag that contains all of
        # the claims
        single_claim_text = _get_text_from_tag(claims[0])
        raw_claims_texts = re.sub(
            r"\s+(\d{1,2}\.\s+[^\d])",  # e.g. " 1. T"
            lambda match: "\n" + match.groups()[0],
            single_claim_text
        ).split("\n")
        claims_texts = [
            re.sub(r"^\d{1,2}\.\s*", "", raw_text)
            for raw_text in raw_claims_texts
            if re.search(r"^\d{1,2}\.\s*", raw_text)
        ]
    
    if add_section_tags:
        claims_texts = [f"[claim] {claim_text}" for claim_text in claims_texts]

    return abstract_text, claims_texts

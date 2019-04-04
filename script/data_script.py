import xml.etree.ElementTree as ET
import pickle, re, sys, os
from lxml import etree
# import xml_parse
from xml.dom import minidom

EVENT_MAP={'None': 0, 'Personnel.Nominate': 1, 'Contact.Phone-Write': 27, 'Business.Declare-Bankruptcy': 3,
           'Justice.Release-Parole': 4, 'Justice.Extradite': 5, 'Personnel.Start-Position': 22,
           'Justice.Fine': 7, 'Transaction.Transfer-Money': 8, 'Personnel.End-Position': 9,
           'Justice.Acquit': 10, 'Life.Injure': 11, 'Conflict.Attack': 12, 'Justice.Arrest-Jail': 13,
           'Justice.Pardon': 14, 'Justice.Charge-Indict': 15, 'Conflict.Demonstrate': 16,
           'Contact.Meet': 17, 'Business.End-Org': 18, 'Life.Be-Born': 19, 'Personnel.Elect': 20, 
           'Justice.Trial-Hearing': 21, 'Life.Divorce': 6, 'Justice.Sue': 23, 'Justice.Appeal': 24,
           'Business.Merge-Org': 32, 'Life.Die': 26, 'Business.Start-Org': 2, 'Justice.Convict': 28,
           'Movement.Transport': 29, 'Life.Marry': 30, 'UNKOWN': 34, 'Justice.Sentence': 31,
           'Justice.Execute': 25, 'Transaction.Transfer-Ownership': 33}

def read_file(xml_path, text_path):
    """
    Read the text property in apf.xml file
    """
    print(xml_path)
    print(text_path)
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()
    
    event_start = {}
    event_end = {}

    event_ident = {}
    event_map = {}
    event = dict()

    # get event information from .apf.xml file
    for events in root.iter("event"):
        ev_type = events.attrib["TYPE"] + "." + events.attrib["SUBTYPE"]
        for mention in events.iter("event_mention"):
            ev_id = mention.attrib["ID"]
            anchor = mention.find("anchor")
            for charseq in anchor:
                start = int(charseq.attrib["START"])
                # +1 is because of half-opened string interval
                end = int(charseq.attrib["END"]) + 1 
                text = re.sub(r"\n", r" ", charseq.text)
                # basic imformation of event
                event_tupple = (ev_type, start, end, text)
                # judge if there is already event mention (correference)
                if event_tupple in event_ident:
                    sys.stderr.write("dulicapte event {}\n".format(ev_id))
                    event_map[ev_id] = event_ident[event_tupple]
                    continue
                event_ident[event_tupple] = ev_id
                event[ev_id] = [ev_id, ev_type, start, end, text]
                event_start[start] = ev_id
                event_end[end] = ev_id

    # transform the .sgm file (.xml file essentially) to string and excluds tags; using minidom instead of ElementTree (better nodes information representing ablility)
    text = ""
    str_empty = ""
    sub = 0
    try:
        doc = minidom.parse(text_path)
    except:
        print("Failed",text_path)
    doc_root = doc.documentElement
    root_children = doc_root.childNodes
    for root_child in root_children: 
        if root_child.nodeName == "#text":
            str_empty = " " * len(root_child.nodeValue)
            text += str_empty
        elif root_child.nodeName == "DOCID" or root_child.nodeName == "DOCTYPE" or root_child.nodeName == "DATETIME":
            sub += len(root_child.childNodes[0].nodeValue)
        elif root_child.nodeName == "BODY": 
            body_children = root_child.childNodes
            for body_child in body_children: 
                if body_child.nodeName == "#text": 
                    str_empty = " " * len(body_child.nodeValue)
                    text += str_empty
                elif body_child.nodeName == "HEADLINE":
                    # substitute whitespaces for HEADLINE
                    str_empty = " " * len(body_child.childNodes[0].nodeValue)
                    text += str_empty
                elif body_child.nodeName == "TEXT":
                    text_children = body_child.childNodes
                    for text_child in text_children:
                        if text_child.nodeName == "#text":
                            text += text_child.nodeValue.replace("\n", " ")
                        elif text_child.nodeName == "TURN":
                            turn_children = text_child.childNodes
                            for turn_child in turn_children:
                                if turn_child.nodeName == "SPEAKER":
                                    text += turn_child.childNodes[0].nodeValue
                                elif turn_child.nodeName == "#text":
                                    text += turn_child.nodeValue.replace("\n", " ")
                        elif text_child.nodeName == "POST":
                            post_children = text_child.childNodes
                            for post_child in post_children:
                                if post_child.nodeName == "POSTER" or post_child.nodeName == "POSTDATE":
                                    str_empty = " " * len(post_child.childNodes[0].nodeValue)
                                    text += str_empty
                                elif post_child.nodeName == "#text":
                                    text += post_child.nodeValue.replace("\n", " ")

    tokens, anchors = read_document(text, sub, event_start, event_end, event_ident, event_map, event)
    return [tokens], [anchors]                                

def read_document(doc, sub, event_start, event_end, event_ident, event_map, event):
    """
    Transform the xml file to tokens and anchors
    """
    tokens = []
    anchors = []
    current = 0

    # In MARKETVIEW_20050206.2009.sgm "&amp;" occurs in HEADLINE and is transformed to "&"
    if "MARKETVIEW_20050206.2009-EV1-1" in event: 
        sub += 4 
    for i in range(len(doc)):

        #Because of the XML transforming of "&amp;" to "&"
        if doc[i] == "&":
            sub += 4

            # these files consider "&amp;" as 1 character
            if "CNN_ENG_20030616_130059.25-EV1-1" in event or "CNN_CF_20030304.1900.04-EV15-1" in event or "BACONSREBELLION_20050226.1317-EV2-1" in event:
                sub -= 4 

        if i+sub in event_start:

            # clear str: e.g. removing special chars; doc[x:y]: half-opened; clean_str will not influence the index.
            new = clean_str(doc[current:i]) 
            tokens += new.split()

            # init anchors with 0 for a special range; '_' serves as a throwaway variable name.
            anchors += [0 for _ in range(len(new.split()))] 
            current = i
            ent = event_start[i+sub]
        if i+sub in event_end: 
            ent = event_end[i+sub]
            if ent == "AGGRESSIVEVOICEDAILY_20041208.2133-EV4-1":
                tokens += ["q&a"]
                anchors += [EVENT_MAP["Contact.Meet"]]
                current = i
            else: 
                new = clean_str(doc[event[ent][2]-sub : event[ent][3]-sub])

                # the statement after "," is the message throwed when condition is false; very useful checking step;
                assert new == event[ent][4] or new == event[ent][4].lower() \
                ,"Error: " + new + " ," + event[ent][4] +" " + str(event[ent][2]-sub) + " " + str(event[ent][3]-sub) 
                tokens += [new.replace(" ", "")]
                anchors += [EVENT_MAP[event[ent][1]]]
                current = i 
    new = clean_str(doc[current : ])
    tokens += new.split()
    anchors += [0 for _ in range(len(new.split()))]
    assert len(tokens) == len(anchors),"sai cmnr"
    return tokens, anchors

def encode_corpus(folder_path):
    file_list = os.path.join(folder_path, "FileList")
    files = []
    with open(file_list) as f:
        for line in f:
            # strip(): remove leading and trailing chars (whitespace default) and return copy of original str
            # split(): split the str into a list based on the chars passed (whitespace default)
            map = line.strip().split()
            if len(map) != 3: continue
            files.append((map[0], map[1].split(",")[-1])) # -1 is the last elemet 
    return files

def read_corpus(folder_path):
    count = 0
    file_list = encode_corpus(folder_path)
    print(file_list)
    tokens, anchors = [], []
    for (file, path) in file_list:
        file_path = os.path.join(folder_path, path, file)
        tok, anc = read_file(file_path + ".apf.xml", file_path + ".sgm")
        count += 1
        tokens += tok
        anchors += anc
    return tokens, anchors

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub("[^A-Za-z0-9().,!?'`-]&", " ", string)  # ^ means matching any char not here
    # string = re.sub("'m", " 'm", string) 
    # string = re.sub("'s", " 's", string) 
    # string = re.sub("'ve", " 've", string) 
    # string = re.sub("n't", " n't", string) 
    # string = re.sub("'re", " 're", string) 
    # string = re.sub("'d", " 'd", string) 
    # string = re.sub("'ll", " 'll", string) 
    string = re.sub("``", " <dot> ", string)
    string = re.sub("''", " <dot> ", string)
    string = re.sub(r"\"", " <dot> ", string) 
    string = re.sub(r"\.", " <dot> ", string)
    string = re.sub(r"\,", " <dot> ", string) 
    string = re.sub(r"!", " <dot> ", string) 
    string = re.sub(r"\(", " <dot> ", string) 
    string = re.sub(r"\)", " <dot> ", string) 
    string = re.sub(r"\?", " <dot> ", string)
    string = re.sub(r"\_", " <dot> ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

if __name__ == "__main__":

    # tk = pickle.load(open("./preprocessing/tokens_bn.bin","rb"))
    # ac = pickle.load(open("./preprocessing/anchors_bn.bin", "rb"))
 
    # tokens type: list, tokens rank: 2, axis 0 indicates each articles, axis 1 indicates the tokens in article
    # anchors type: list, anchors rank: 2, axis 0 indicates each articles, axis 1 indicates the anchor information of corresponding tokens in article

    tokens_bn, anchors_bn  = read_corpus("./script/ACE2005ENG/orig/bn")
    pickle.dump(tokens_bn, open("./preprocessing/tokens_bn.bin","wb"))
    pickle.dump(anchors_bn, open("./preprocessing/anchors_bn.bin", "wb"))
    
    tokens_nw, anchors_nw  = read_corpus("./script/ACE2005ENG/orig/nw")
    pickle.dump(tokens_nw, open("./preprocessing/tokens_nw.bin","wb"))
    pickle.dump(anchors_nw, open("./preprocessing/anchors_nw.bin", "wb"))

    tokens_bc, anchors_bc = read_corpus("./script/ACE2005ENG/orig/bc")
    pickle.dump(tokens_bc, open("./preprocessing/tokens_bc.bin","wb"))
    pickle.dump(anchors_bc, open("./preprocessing/anchors_bc.bin", "wb"))
    
    tokens_cts, anchors_cts = read_corpus("./script/ACE2005ENG/orig/cts")
    pickle.dump(tokens_cts, open("./preprocessing/tokens_cts.bin","wb"))
    pickle.dump(anchors_cts, open("./preprocessing/anchors_cts.bin", "wb"))

    tokens_wl, anchors_wl = read_corpus("./script/ACE2005ENG/orig/wl")
    pickle.dump(tokens_wl, open("./preprocessing/tokens_wl.bin","wb"))
    pickle.dump(anchors_wl, open("./preprocessing/anchors_wl.bin", "wb"))

    print("1")
    



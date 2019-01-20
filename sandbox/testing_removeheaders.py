from sklearn.datasets import fetch_20newsgroups
import re

subset = 'test'
shuffle = True
random_state = 42
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']

raw_documents = fetch_20newsgroups(subset=subset,
                                   categories=categories,
                                   shuffle=shuffle,
                                   random_state=random_state)

########################################################################################################################
## Taken from twenty_newsgroups.py


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

##
########################################################################################################################


remove = ['headers']
if 'headers' in remove:
    raw_documents.data = [strip_newsgroup_header(text) for text in raw_documents.data]
if 'footers' in remove:
    raw_documents.data = [strip_newsgroup_footer(text) for text in raw_documents.data]

print(raw_documents.data[0])

# headless_docs = []
# for doc in raw_documents.data:
#     headless_docs.append(remove_headers(doc))
#
# raw_documents = fetch_20newsgroups(subset=subset,
#                                    categories=categories,
#                                    shuffle=shuffle,
#                                    random_state=random_state,
#                                    remove=('headers',))
# # print(raw_documents.data[1])
#
# print(headless_docs[0])
# print('-*'*20)
# print(raw_documents.data[0])

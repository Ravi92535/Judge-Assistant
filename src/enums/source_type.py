from enum import Enum

class SourceType(Enum):
    FIR = "FIR"
    NATIONAL_DOC = "NationalDoc"


'''
doc = DocType.FIR

print(doc)        # DocType.FIR
print(doc.name)   # FIR
print(doc.value)  # FIR

'''
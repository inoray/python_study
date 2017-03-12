from xml.etree.ElementTree import Element, SubElement, dump, ElementTree, parse

"""
다음과 같은 구조의 XML문서를 생성해 보자
<note date="20120104">
    <to>Tove</to>
    <from>Jani</from>
    <heading>Reminder</heading>
    <body>Don't forget me this weekend!</body>
</note>
"""

note = Element("note")
note.attrib["date"] = "20120104"

to = Element("to")
to.text = "Tove"
note.append(to)

SubElement(note, "from").text = "Jani"
SubElement(note, "heading").text = "Reminder"
SubElement(note, "body").text = "Don't forget me this weekend!"


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


indent(note)
ElementTree(note).write("note.xml")

dump(note)


"""
XML 문서를 파싱해 보자
"""

tree = parse("note.xml")
note2 = tree.getroot()

# 어트리뷰트 값 읽기
print(note2.get("date"))
print(note2.get("foo", "default"))
print(note.keys())
print(note.items())

from_tag = note.fine("from")




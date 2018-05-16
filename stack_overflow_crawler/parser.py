"""
* 任意のタグの削除
* タグ名を削除し、一つのQ or Aを一行にまとめる
"""

from bs4 import BeautifulSoup
from sys import exit

#REMOVE_TAGS = ["code"]
REMOVE_TAGS = ["code", "qid"]

with open("question.txt") as q:
    qsoup = BeautifulSoup(q, "lxml")

with open("answer.txt") as a:
    asoup = BeautifulSoup(a, "lxml")

for soup in (qsoup, asoup):
    for tag_name in REMOVE_TAGS:
            [tag.decompose() for tag in soup.find_all(tag_name)]

q_txt = open("Q_data.txt", "w")
a_txt = open("A_data.txt", "w")
print(len(qsoup.find_all("post")))
for q in qsoup.find_all("post"):
    txt = "".join(q.find_all(string=True))
    txt = txt.replace("\n", " ").strip()
    q_txt.write(txt + "\n")

for a in asoup.find_all("post"):
    txt = "".join(a.find_all(string=True))
    txt = txt.replace("\n", " ").strip()
    a_txt.write(txt + "\n")

q_txt.close(); a_txt.close()

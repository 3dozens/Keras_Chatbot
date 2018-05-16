import urllib.request as request
import json
from sys import exit
from gzip import decompress
from tqdm import tqdm

"""
htmlに適合したテキストに独自タグ<question>, <qid>を加えたデータを吐き出します。
htmlには適合していませんが、beautifulsoup上で扱いやすいのでこの形にしました。
beautifulsoup上では、独自タグを標準のhtmlタグと同様に扱えます。(たぶん)
"""

PAGE_NUMBER = 200

fQ = open("question.txt", "w")
fA = open("answer.txt", "w")

err_cnt = 0
for page in tqdm(range(1, PAGE_NUMBER + 1)):
    # answer取ってくる
    answer_compressed = request.urlopen("https://api.stackexchange.com/2.2/answers\
?order=desc&sort=votes&site=stackoverflow&filter=withbody&pagesize=100&page={}\
&access_token=PP59e0Edm*uhNWQwsWimNg))&key=QHB7XmweW2lQSEdOlTQRbw(("\
    .format(page)).read()
    answer_json_str = decompress(answer_compressed).decode("utf-8")
    answer = json.loads(answer_json_str)

    # questionを取ってくる
    question_ids = [str(ans["question_id"]) for ans in answer["items"]]
    query_ids = ";".join(question_ids)

    question_compressed = request.urlopen("https://api.stackexchange.com/2.2/questions/{}\
?order=desc&sort=activity&site=stackoverflow&filter=withbody\
&access_token=PP59e0Edm*uhNWQwsWimNg))\
&key=QHB7XmweW2lQSEdOlTQRbw((".format(query_ids)).read()
    question_json_str = decompress(question_compressed).decode("utf-8")
    question = json.loads(question_json_str)

    # ファイルに吐き出す
    ## answer と question を対応付ける
    q_a = {}
    for ans in answer["items"]:
        for q in question["items"]:
            if ans["question_id"] == q["question_id"]:
                try:
                    prefix = "<qid>" + str(ans["question_id"]) + "</qid>"
                    q_a[prefix + q["body"]] = prefix + ans["body"]
                except Exception as e:
                    err_cnt += 1
                    print("error count =", err_cnt)
                    print(e)
                    print("-----q-----")
                    print(q)
                    print("-----a-----")
                    print(ans)

                break

    for q, a in q_a.items():
        fQ.write("<post>" + q + "</post>\n")
        fA.write("<post>" + a + "</post>\n")

fQ.close(); fA.close()

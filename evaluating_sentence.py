import os

def process_meta(meta_path, list_unuse=[]):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            # print(line)
            n, t = line.strip('\n').split('|')

            name.append(n)
            text.append(t)
        return name, text

class EvalMeanFrequencyScore:
    def __init__(self, path='train.txt', dict_path='syllable_g2p.txt', low_count=100, high_count=1000):
        dict_count = {}
        basename, text = process_meta(os.path.join(path), [])
        for t in text:
            t = t[1:-1]
            tmp = t.split()
            for i in tmp:
                dict_count[i] = dict_count.get(i, 0) + 1

        dict_count = {k: dict_count[k] for k in sorted(dict_count, key=dict_count.get, reverse=True)}
        list_count = [k for k, v in dict_count.items() if v < low_count]
        list_h_c = [k for k, v in dict_count.items() if v > high_count]
        # print(dict_count)
        self.list_l_word = []
        self.list_h_word = []


        with open(dict_path, 'r', encoding='utf8') as rf:
            lines = rf.read().split('\n')
            for line in lines:
                check_total = False
                temp = line.split()
                word = temp[0]
                phonme = temp[1:]
                for p in phonme:
                    if p in list_count:
                        self.list_l_word.append(word)
                        check_total = True
                        break
                if check_total:
                    continue
                check = True
                for p in phonme:
                    if p not in list_h_c:
                        check = False
                        break
                if check:
                    self.list_h_word.append(word)
                    continue


        # print(len(self.list_h_word))
        # print(list_h_word)

        # print(len(self.list_l_word))
        # print(list_l_word)

    def cal(self, sentence):
        total = 0
        list_word = sentence.split()
        dem1 = 0
        dem2 = 0
        dem3 = 0
        for word in list_word:
            if word in self.list_h_word:
                # print('1', end=' ; ')
                total += 3
                dem1 += 1
            elif word in self.list_l_word:
                # print('-1', end=' ; ')
                total += 0.5
                dem2 += 1
            else:
                total += 1.5
                dem3 += 1
        print(dem1, dem2, dem3)
        return 1.0 * total / len(list_word) / 3

eval = EvalMeanFrequencyScore()
# sentence = 'ủy ban xã cho biết khó để xác định vụ việc nguy hiểm ra sao'
sentence = 'anh ấy đang rất cần nó'
print(eval.cal(sentence))



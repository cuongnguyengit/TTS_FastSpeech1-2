from collections import defaultdict

class G2p:
    def __init__(self, path_dict):
        self.dict_word_char = defaultdict(list)
        self.dict_w2p = {}
        with open(path_dict, 'r', encoding='utf8') as rf:
            lines = rf.read().split('\n')
            for line in lines:
                temp = line.split()
                word = temp[0]
                phonme = ' '.join(temp[1:])
                self.dict_w2p[word] = phonme
                if word not in self.dict_word_char[word[0]]:
                    self.dict_word_char[word[0]].append(word)

    def g2p(self, sentence):
        list_word = sentence.split()
        for i, word in enumerate(list_word):
            if word in self.dict_word_char[word[0]]:
                list_word[i] =  self.dict_w2p[word]
            else:
                list_word[i] = 'spn'
        return ' '.join(list_word)


if __name__ == '__main__':
    gen = G2p('syllable_g2p.txt')
    print(gen.g2p('xin chào tùng xê tê'))
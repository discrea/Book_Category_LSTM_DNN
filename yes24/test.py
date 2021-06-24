import MeCab as mc
# m = mc.Tagger()
# te = m.parse('아버지가방에들어가신다')
# mo = m.set_morphs(te)
# print(te)
# print(mo)

from konlpy.tag import Mecab
tokenizer = Mecab()
print(tokenizer.morphs('아버지가방에들어가신다'))
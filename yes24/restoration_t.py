from ko_restoration import main

mecab, complex_verb_set = main.set_env()

lines = ['원래 인공지능을 공부하려고 했을 때']
         # '자연어 처리에 대해 관심이 많지 않았습니다',
         # '인물 인식, 모션트래킹등 영상분야에 대한 것에 만이 관심이 있었거든요.',
         # '사실 메인은 주식 시계열 분석 및 예측.',
         # '그러다 RNN LSTM 실습을 같이 하던 조원에게서 자연어 처리에 대한 영감을 받고 관심이 생겼습니다.',
         # ]

print('원형 복원 전 : \n', lines)
result = main.start_restoration(mecab, complex_verb_set, lines)
print('원형 복원 후 : \n', result)
list_result = result.split()
print(list_result)

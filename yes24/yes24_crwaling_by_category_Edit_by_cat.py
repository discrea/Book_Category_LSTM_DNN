import time
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

category = ['건강 취미', '경제 경영', '사회 정치', '소설 시 희곡', '수험서 자격증', '에세이', '역사', '인문',
            '인물', '자기계발', '자연과학', '잡지', '종교']
category_num = ['001001011', '001001025', '001001022', '001001046', '001001015',
                '001001047', '001001010', '001001019', '001001020', '001001026',
                '001001002', '001001024', '001001021']
category_sub_num = [['008', '003', '021', '017', '016'], ['007', '008', '010', '0009', '001'],
                    ['008', '009', '005', '015'], ['011', '012', '013', '014', '001', '002', '003'],
                    ['010', '004', '019', '017', '023', '013'], ['003', '010', '006', '002', '001'],
                    ['002', '005', '006', '007', '008', '012'], ['001', '004', '003', '015', '014'],
                    ['008', '009', '002', '004'], ['014', '022', '006', '015'], ['003', '001', '004']]

book_name = []
book_medium_category = []
book_small_category = []
book_summary = []

url_format = 'http://www.yes24.com/24/category/bestseller?CategoryNumber={}&sumgb=06&PageNumber={}'

for i in range(len(category)): # 중분류
    num = 0
    for j in range(len(category_sub_num[i])): #소분류
        for k in range(1, 51): # 페이지 최소/최대
            url_craw = url_format.format(category_num[i]+category_sub_num[i][j], k)
            html = requests.get(url_craw)  # html로 가져오고
            soup = BeautifulSoup(html.text, 'html.parser')  # bs4로 읽어서
            book_list = soup.select('.goodsTxtInfo')  # 페이지 당 책 list
            if not book_list: # 페이지가 비어있으면 for break
                break
            for l in range(len(book_list)):
                book_info = book_list[l]
                title = book_info.select_one('p > a').get_text() # 책 제목
                link_part = book_info.select_one('p > a')['href']  # 해당책의 링크 조각을 가져온 후
                book_url = 'http://www.yes24.com' + link_part  # 책의 링크로 들어가서
                book_html = requests.get(book_url)  # 들어가서
                book_soup = BeautifulSoup(book_html.text, 'html.parser')  # 이제 진짜 들어가서
                book_category_soup = book_soup.select('.yesAlertLi > li > a')  # 카테고리 관련 테그 긁어오고
                category_sub = []
                for i in range(5):  # 카테고리가 대, 중, 소 분류로 나뉘어 있다
                    if book_category_soup[i].get_text() == '국내도서':
                        for cat in book_category_soup[i:i + 3]:
                            category_sub.append(cat.get_text())  # 반복문으로 하나씩 가져오기
                book_summary_soup = book_soup.select('.infoWrap_txtInner')  # 책 소개 및 줄거리 부분
                text = ''
                for summary in book_summary_soup:  # 여러 br태그에 쪼개져서 들어있다
                    text += summary.getText()
                book_name.append(title)                         # 책 제목
                book_medium_category.append(category_sub[0])    # 중 분류
                book_small_category.append(category_sub[1])     # 소 분류
                book_summary.append(text)                       # 책 소개
                num += 1
                print(num, category_sub, '', title)

raw_data = pd.DataFrame({'Title':book_name,
                         'Medium_category':book_medium_category,
                         'Small_category':book_small_category,
                         'Introduction':book_summary
                         })
raw_data.to_csv('./crawling_data/project_test_data.csv')





# -*- coding: utf-8 -*-
"""yes24_crwaling_by_category.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LlnHrPFCzDmFFhO7oUCzE12Nf3-53qT_

1. 건강 취미
2. 경제 경영
3. 사회 정치
4. 소설 시 희곡
5. 수험서 자격증
6. 에세이
7. 역사
8. 인문
9. X인X물X. => 데이터 적음
10. 자기계발
11. 자연과학
12. 잡지. => 데이터 적음
13. 종교
"""

import os
import re
import time
import requests
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from urllib.request import urlopen

def get_category(url = 'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011008&sumgb=06&PageNumber=35',
             headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'}):

    html = requests.get(url, headers= headers)      # html로 가져오고
    soup = BeautifulSoup(html.text, 'html.parser')  # bs4로 읽어서
    book_info = soup.select_one('.goodsTxtInfo')    # 아무거나 책 찍고 
    link_part = book_info.select_one('p > a')['href']   # 해당책의 링크 조각을 가져온 후

    book_url = 'http://www.yes24.com' + link_part                   # 책의 링크로 들어가서
    book_html = requests.get(book_url, headers= headers)            # 들어가서
    book_soup = BeautifulSoup(book_html.text, 'html.parser')        # 이제 진짜 들어가서

    book_category_soup = book_soup.select('.yesAlertLi > li > a')   # 카테고리 관련 테그 긁어오고
    category_sub = []
    for i in range(5):                       # 카테고리가 대, 중, 소 분류로 나뉘어 있다
        if book_category_soup[i].get_text() == '국내도서':
            for cat in book_category_soup[i:i+3]:
                category_sub.append(cat.get_text())                   # 반복문으로 하나씩 가져오기
    
    return category_sub

# for url in url_list[2:7]:
#     print(get_category(url))

Large_category = []
Medium_category = []
Small_category = []
page = []
Url_format = []

url_list = [
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011008&sumgb=06&PageNumber=35',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011003&sumgb=06&PageNumber=24',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011021&sumgb=06&PageNumber=24',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011017&sumgb=06&PageNumber=30',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001011016&sumgb=06&PageNumber=24',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001025007&sumgb=06&PageNumber=37',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001025008&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001025010&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001025009&sumgb=06&PageNumber=40',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001025001&sumgb=06&PageNumber=46',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001022008&sumgb=06&PageNumber=35',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001022009&sumgb=06&PageNumber=27',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001022005&sumgb=06&PageNumber=36',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001022015&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046011&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046012&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046013&sumgb=06&PageNumber=24',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046014&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046001&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046002&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001046003&sumgb=06&PageNumber=35',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015010&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015004&sumgb=06&PageNumber=29',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015019&sumgb=06&PageNumber=31',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015017&sumgb=06&PageNumber=21',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015023&sumgb=06&PageNumber=25',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001015013&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001047003&sumgb=06&PageNumber=10',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001047010&sumgb=06&PageNumber=12',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001047006&sumgb=06&PageNumber=18',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001047002&sumgb=06&PageNumber=37',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001047001&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010002&sumgb=06&PageNumber=18',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010005&sumgb=06&PageNumber=39',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010006&sumgb=06&PageNumber=17',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010007&sumgb=06&PageNumber=16',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010008&sumgb=06&PageNumber=13',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001010012&sumgb=06&PageNumber=34',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001019001&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001019004&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001019003&sumgb=06&PageNumber=21',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001019015&sumgb=06&PageNumber=20',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001019014&sumgb=06&PageNumber=21',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001026008&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001026009&sumgb=06&PageNumber=18',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001026002&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001026004&sumgb=06&PageNumber=15',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001002014&sumgb=06&PageNumber=27',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001002022&sumgb=06&PageNumber=11',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001002006&sumgb=06&PageNumber=30',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001002015&sumgb=06&PageNumber=18',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001021003&sumgb=06&PageNumber=50',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001021001&sumgb=06&PageNumber=14',
'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001021004&sumgb=06&PageNumber=36'
]

for url in url_list:
    category = get_category(url)
    Large_category.append(category[0])
    Medium_category.append(category[1])
    Small_category.append(category[2])
    page.append(url[-2:])
    Url_format.append(url[:-2])
    print(category, url[-2:], url[:-2])

category = pd.DataFrame({'Large_category':Large_category,
                         'Medium_category':Medium_category,
                         'Small_category':Small_category,
                         'page':page,
                         'Url_format':Url_format
                         })
category

category.page = category.page.astype(int)
category.info()

grouped = category[['Medium_category','page']].groupby(category['Medium_category'])
grouped.sum()

for temp in category.iloc[:5]:
    print(url)

book = pd.DataFrame()
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'}

book_name = []
# book_link = []
book_medium_category = []
book_small_category = []
book_summary = []

for j in range(len(category)):                              # 노가다한 링크 개수만큼 순회
    for i in range(1,category.iloc[j]['page']+1):           # 소주재 별 순회

        
        url = category.iloc[j]['Url_format'].format(i)                      # yes24 베스트셀러 페이지 url
        html = requests.get(url, headers= headers)      # html로 가져오고
        soup = BeautifulSoup(html.text, 'html.parser')  # bs4로 읽어서
        # print(soup)
        title_tags = soup.select('.goodsTxtInfo')       # 한 페이지에 ??개의 책 정보를 리스트로 가져온다
        
        for book_info in title_tags:                            # 리스트를 순회하며
            title = book_info.select_one('p > a').get_text()    # 책 제목
            link_part = book_info.select_one('p > a')['href']   # 해당책의 링크 조각을 가져온 후

        # test 할 때는 한칸 내리고
            book_url = 'http://www.yes24.com' + link_part                   # 책의 링크로 들어가서
            book_html = requests.get(book_url, headers= headers)            # 들어가서
            book_soup = BeautifulSoup(book_html.text, 'html.parser')        # 이제 진짜 들어가서

            book_category_soup = book_soup.select('.yesAlertLi > li > a')   # 카테고리 관련 테그 긁어오고
            category_sub = []
            for i in range(6):                                          # 카테고리가 대, 중, 소 분류로 나뉘어 있다
                if book_category_soup[i].get_text() == '국내도서':
                    for cat in book_category_soup[i+1:i+3]:
                        category_sub.append(cat.get_text())             # 반복문으로 하나씩 가져오기

            book_summary_soup = book_soup.select('.infoWrap_txtInner')      # 책 소개 및 줄거리 부분
            text = ''
            for summary in book_summary_soup:                               # 여러 br태그에 쪼개져서 들어있다
                text += summary.getText()


            book_name.append(title)
            book_medium_category.append(category_sub[0])
            book_small_category.append(category_sub[1])
            book_summary.append(text)

            print('\n', '@'*40, '\n')
            print('book title :', title, '\n', '='*40)
            print('link part :', link_part, '\n', '='*40)
            print('link url :', book_url, '\n', '='*40)
            print('book category :', category_sub, '\n', '='*40)
            print('book summary :', text, '\n', '='*40)

raw_data = pd.DataFrame({'Title':book_name,
                         'Medium_category':book_medium_category,
                         'Small_category':book_small_category,
                         'Introduction':book_summary
                         })
raw_data

temp



for temp in category.iloc[:5]:
    print(temp[0])

category

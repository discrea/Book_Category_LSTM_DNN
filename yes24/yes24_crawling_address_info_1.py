"""
# Yes24 웹 베스트샐러 문류별 링크 추출 #

yes24의 분류 각각의 책 내용 클롤링은 편하나
링크는 규칙성을 발견하지 못해 수작업을 하다 빡쳐서 만들어봤다
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


"""
예측컨데 주소의 CategoryNumber는 
0010010000000 ~ 001001099999 사이로 추론할 수 있다.
내가 노가다 할순 없으니 cpu에 채찍질을 한다
"""

"""
5시간의 밤샘 코딩중 너무 무식하게 했음을 깨닫는다
뒤엎고 다시
"""
# 주소
url_format = 'http://www.yes24.com/24/category/bestseller?CategoryNumber=0010010{:0>5}&sumgb=06&PageNumber=1'
# 나 콤퓨타 아니다. 나 휴먼이다.
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'}

total_medium_category = []
total_small_category = []
total_category_number = []
total_page_amount = []
medium_category = []
small_category = []
category_number = []
page_amount = []
error_page = []
true_flag = False
t = 0
e = False
start = 0
end = 99999
for i in range(start, end+1):     # 일해라 컴퓨터
    if i % 100 == 0:
        medium_category = []
        small_category = []
        category_number = []
        page_amount = []
        t = 0
        print('{:0>5} ~ {:0>5} : '.format(i, i+99), end='')


    url = url_format.format(i)
    html = requests.get(url, headers=headers)  # html로 가져오고
    soup = BeautifulSoup(html.text, 'html.parser')  # bs4로 읽어서
    random_book = soup.select_one('.goodsTxtInfo')  # 아무거나 책 찍고
    if not random_book:print('F', end='')
    else:
        try:
            regex = re.compile('(\d+)(?!.*\d)')             # 마지막 페이지를 가져오기 위한 정규식

            last_page_link = soup.select_one('.page > img').select('.hover')[1]['href']
            last_page = regex.findall(last_page_link)[0]    # 패턴과 매칭되는 문자를 리스트로 반환

            link_part = random_book.select_one('p > a')['href']         # 해당책의 링크 조각을 가져온 후
            book_url = 'http://www.yes24.com' + link_part               # 책의 링크로 들어가서
            book_html = requests.get(book_url, headers=headers)         # 들어가서
            book_soup = BeautifulSoup(book_html.text, 'html.parser')    # 이제 진짜 들어가서

            book_category_soup = book_soup.select('.yesAlertLi > li > a')  # 카테고리 관련 테그 긁어오고
            for j in range(5):  # 카테고리가 대, 중, 소 분류로 나뉘어 있다
                if book_category_soup[j].get_text() == '국내도서':
                    medium_category.append(book_category_soup[j+1].get_text())
                    small_category.append(book_category_soup[j+2].get_text())
                    category_number.append('{:0>5}'.format(i))
                    page_amount.append(last_page)

                    print('T', end='')
                    t += 1
                    true_flag = True
                    break
            if not true_flag:
                true_flag = False
                print('O', end='')      # 국내도서가 아닐 경우


        except IndexError :
            e = True
            error_page.append('{:0>5}'.format(i))
            print('E', end='')

    if i % 100 == 99:
        print(' || Exist Page = ',t,end='')
        if t != 0:
            info = pd.DataFrame({'medium_category' : medium_category,
                                 'small_category': small_category,
                                 'category_number': category_number,
                                 'page_amount': page_amount
                                 })
            info.to_csv('../data_backup/{:0>5}_{:0>5}'.format(i-99, i))
            total_medium_category += medium_category
            total_small_category += small_category
            total_category_number += category_number
            total_page_amount += page_amount
            medium_category = []
            small_category = []
            category_number = []
            page_amount = []

        if e:
            print(' || Error Occured On = ', error_page)
            error_page = []
            e = False
        else:print('')


result = pd.DataFrame({'medium_category' : total_medium_category,
                     'small_category': total_small_category,
                     'category_number': total_category_number,
                     'page_amount': total_page_amount
                     })
info.to_csv('../data_backup/address_info')



from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from utils import clear_folder
import requests
import json
import pandas as pd
import re
import csv

#爬虫类
class Spider:
    def __init__(self, url, timeout):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.header = {
            #写自己的cookie和header
        }
        self.params = {
            "itemId": "",
            "sellerId": "123456789",
            "currentPage": "1",
            "callback": "jsonp723"
        }
        self.df = pd.DataFrame()#传递数据用的

    #获取页面
    def get_page(self):
        try:
            page = requests.get(self.url, headers=self.header, timeout=self.timeout)
            page.raise_for_status()
            with open('temp/data/page.txt', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(page.text)
            print("获取页面信息成功")
            return page.text
        except requests.exceptions.RequestException as err:
            print("获取页面信息失败：", str(err))
            return None

    #接口函数，方便在UI中调用
    def Interface(self, platform):
        page = self.get_page()
        if page is None:
            return '页面为空', None
        try:
            if platform == 'taobao':
                self.df = self.taobao_prase_page(page)
            elif platform == 'jingdong':
                self.df = self.jingdong_prase_page(page)
            return "保存DataFrame成功", self.df
        except Exception as err:
            return str(err) + '获取失败', None

    #获取评论的接口
    def get_comments(self, platform):
        if platform == 'taobao':
            executor = ThreadPoolExecutor(max_workers=48)
            clear_folder('temp/TB_comment')
            for index in self.df.index:
                item_id = self.df.at[index, 'ID']
                writer = open(f"temp/TB_comment/{item_id}comments.csv", "w", newline='', encoding='utf-8-sig')
                executor.submit(self.tbcomments_crawl, item_id, 20, writer)
            executor.shutdown(wait=True)
        elif platform == 'jingdong':
            executor = ThreadPoolExecutor(max_workers=30)
            clear_folder('temp/JD_comment')
            for index in self.df.index:
                item_id = self.df.at[index, 'ID']
                writer = open(f"temp/JD_comment/{item_id}comments.csv", "w", newline='', encoding='utf-8-sig')
                executor.submit(self.jdcomments_crawl, item_id, 5, writer)
            executor.shutdown(wait=True)

    #处理淘宝页面的信息
    def taobao_prase_page(self, page):
        data_list = []
        page_info = re.findall('g_page_config = (.*);', page)[0]
        page_info = json.loads(page_info)
        info = page_info['mods']['itemlist']['data']['auctions']

        for index in info:
            level_classes = index['shopcard']['levelClasses']
            seller_credit = level_classes[0].get('sellerCredit') if level_classes else None
            total_rate = level_classes[0].get('totalRate') if level_classes else None
            data_dict = {
                "ID": index['nid'],
                "商品": index['raw_title'],
                "价格": index['view_price'],
                "店铺": index['nick'],
                "店铺地址": index['shopLink'],
                "购买人数": index['view_sales'],
                "发货地": index['item_loc'],
                "详情页": index['detail_url'],
                "图片": index['pic_url'],
                "店铺信息": {
                    '天猫' if level_classes and any(
                        d.get('levelClass') == 'icon-supple-level-jinguan' for d in level_classes) else '无',
                    seller_credit,
                    total_rate
                }
            }
            data_list.append(data_dict)
        pd.DataFrame(data_list).to_csv('temp/data/TBdata.csv', header=True)
        print("成功缓存淘宝数据")
        return pd.DataFrame(data_list)

    #处理京东页面的信息
    def jingdong_prase_page(self, page):
        data_list = []
        data = BeautifulSoup(page, 'html.parser')
        infos = data.find_all('li', class_='gl-item')


        for info in infos:
            # print(info)
            ID = info.get('data-sku')
            item = info.find('div', class_='p-name p-name-type-2').get_text(strip=True)
            price = info.find('i').get_text(strip=True)
            try:
                shop = info.find('a', class_='curr-shop' if 'curr-shop' in info else 'curr-shop hd-shopname').get_text(
                    strip=True)
            except:
                shop = None
            try:
                shoplink = info.find('div', class_='p-shop').a['href']
            except:
                shoplink = None
            img = info.find('img').get('data-lazy-img')
            detail_page = info.find('div', class_='p-name p-name-type-2').a['href']
            loc = info.find('div', class_='p-stock')['data-province']
            try:
                tips = info.find('i', class_='goods-icons J-picon-tips J-picon-fix').get_text(strip=True)
            except Exception as e:
                tips = '无'
            try:
                shopreputation = info.find('div', class_='p-shop')['data-reputation']
            except Exception as e:
                shopreputation = 0

            data_dict = {
                'ID': ID,
                '商品': item,
                '价格': price,
                '店铺': shop,
                '店铺地址': shoplink,
                "购买人数": None,
                "发货地": loc,
                '详情页': detail_page,
                '图片': img,
                '店铺信息': {'标签': tips, '店铺信誉': shopreputation}
            }
            data_list.append(data_dict)

        pd.DataFrame(data_list).to_csv('temp/data/JDdata.csv', header=True)
        print("成功缓存京东数据")
        return pd.DataFrame(data_list)

    #获取淘宝评论
    def tbcomments_crawl(self, itemId, pages, writer):
        self.header['referer'] = 'https://detail.tmall.com/item.htm?id=' + itemId
        tbcom_url = "https://rate.tmall.com/list_detail_rate.htm"
        attris = ["rateContent"]
        csv_writer = csv.writer(writer)
        csv_writer.writerow(attris)
        print('正在获取评论. . . . . .\n')
        for i in range(pages):
            page = i + 1
            params = {
                "itemId": itemId,
                "sellerId": "123456789",
                "currentPage": str(page),
                "callback": "jsonp723"
            }
            req = requests.get(tbcom_url, params, headers=self.header).content.decode('utf-8')[11:-1]
            # print(req)
            result = json.loads(req)
            comments = result["rateDetail"]["rateList"]
            if len(comments) == 0:
                print("获取评论失败")
            for comment in comments:
                tmp = []
                # print(comment)
                for attri in attris:
                    tmp.append(comment[attri])
                csv_writer.writerow(tmp)

    #获取京东评论
    def jdcomments_crawl(self, itemId, pages, writer):
        base_url = 'https://club.jd.com/comment/productPageComments.action'
        callback_param = 'fetchJSON_comment98'
        score_param = '0'
        sort_type_param = '5'
        page_size_param = '10'
        is_shadow_sku_param = '0'
        fold_param = '1'

        comments = []
        scores = []
        print('正在获取评论. . . . . .\n')
        for page in range(1, pages, 1):
            url = f'{base_url}?callback={callback_param}&productId={itemId}&score={score_param}&sortType={sort_type_param}&pageSize={page_size_param}&isShadowSku={is_shadow_sku_param}&fold={fold_param}&page={page}'
            response = requests.get(url, headers=self.header)
            if response == None:
                print("获取评论失败")
            data = response.text
            json_data = json.loads(data[data.find('(') + 1:-2])
            comments_data = json_data.get("comments", [])
            for comment in comments_data:
                comments.append(comment.get("content", "").strip().replace("\n", ""))
                scores.append(comment.get("score", ""))

        csv_writer = csv.writer(writer)
        attris = ["评论"]
        csv_writer.writerow(attris)
        for comment, score in zip(comments, scores):
            csv_writer.writerow([comment])

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QComboBox, QLabel, QHBoxLayout, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt5.QtGui import QFont
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from Evalution import CommentEvaluator
from utils import download_image, clear_folder
import Spider
import os
import shutil
import time
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))


# 四个线程
class SpiderThread(QThread):
    signal = pyqtSignal(str, pd.DataFrame)

    def __init__(self, url, platform):
        QThread.__init__(self)
        self.url = url
        self.platform = platform
        self.spider = None
        self.executor = None

    def run(self):
        self.spider = Spider.Spider(self.url, 5)
        self.evaluator = CommentEvaluator('Model/model/SA_zh.pt', 'Model/model/word2index.pt',
                                          'Model/model/index2word.pt')

        status, data = self.spider.Interface(self.platform)
        self.signal.emit(status, data)
        self.spider.get_comments(self.platform)
        self.evaluator.Interface(self.platform)


class EvalutionThread(QThread):
    signal = pyqtSignal(str, object, int)

    def __init__(self, platform):
        QThread.__init__(self)
        self.platform = platform

    def run(self):
        print("开始评价")
        if self.platform == 'jingdong':
            data = pd.read_csv('temp/data/JDdata.csv', header=0)
            if 'score_normalized' not in data.columns:
                data = None
            status = "评价失败,请重试"
        elif self.platform == 'taobao':
            data = pd.read_csv('temp/data/TBdata.csv', header=0)
            if 'score_normalized' not in data.columns:
                data = None
            status = "评价失败，请重试"
        else:
            data = None
            status = "评价失败，请重试"
        print("评价中. . . . . .")
        self.signal.emit(status, data, 1)


class SaveThread(QThread):
    signal = pyqtSignal(str)

    def __init__(self, platform):
        QThread.__init__(self)
        self.platform = platform

    def run(self):
        print("保存中...")
        now = str(time.time())
        try:
            if self.platform == 'jingdong':
                print('copying')
                shutil.copyfile('temp/data/JDdata.csv', 'data/JDdata' + now + '.csv')
                print('copy success')
                status = '保存成功，路径为data/JDdata' + now + '.csv'
            elif self.platform == 'taobao':
                shutil.copyfile('temp/data/TBdata.csv', 'data/TBdata' + now + '.csv')
                status = '保存成功，路径为data/TBdata' + now + '.csv'
            else:
                print("other")
                status = "保存失败!"
            self.signal.emit(status)
        except Exception as e:
            self.signal.emit(e)


class BackThread(QThread):
    signal = pyqtSignal(str, object, int)

    def __init__(self, platform):
        QThread.__init__(self)
        self.platform = platform

    def run(self):
        status = "返回失败，请重试"
        if self.platform == 'jingdong':
            data = pd.read_csv('temp/data/JDdata.csv', header=0)
            if 'score_normalized' not in data.columns:
                self.signal.emit(status, data, 0)
            else:
                self.signal.emit(status, data, 1)
        elif self.platform == 'taobao':
            data = pd.read_csv('temp/data/TBdata.csv', header=0)
            if 'score_normalized' not in data.columns:
                self.signal.emit(status, data, 0)
            else:
                self.signal.emit(status, data, 1)
        else:
            data = None
            self.signal.emit(status, data, 0)

#UI类
class App(QWidget):
    # 初始化信号！！！！不初始化等着被卡死吧
    search_complete_signal = pyqtSignal(str, pd.DataFrame)
    auto_evaluate_complete_signal = pyqtSignal(str, object, int)
    save_data_complete_signal = pyqtSignal(str)
    back_complete_signal = pyqtSignal(str, object, int)

    def __init__(self):
        super().__init__()
        self.title = '超级智能的商品推荐系统--by Alcatraz'
        self.initUI()
        self.platform_dict = {
            '京东': 'jingdong',
            '淘宝': 'taobao'
        }

        self.search_complete_signal.connect(self.update_result)
        self.auto_evaluate_complete_signal.connect(self.update_result)
        self.save_data_complete_signal.connect(self.show_message)
        self.back_complete_signal.connect(self.update_result)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(500, 200, 800, 600)

        layout = QVBoxLayout()

        # Title
        font = QFont("Arial", 20, QFont.Bold)
        self.label = QLabel('智能商品推荐系统')
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignHCenter)
        self.label.setStyleSheet('QLabel { max-height: 40px; }')
        layout.addWidget(self.label)

        # 输入框
        self.inputLine = QLineEdit()
        self.inputLine.setFixedSize(750, 30)
        self.inputLine.setPlaceholderText("请输入商品名称")
        layout.addWidget(self.inputLine)

        # 按钮
        buttonsLayout = QHBoxLayout()
        self.comboBox = QComboBox()
        self.comboBox.addItem("京东")
        self.comboBox.addItem("淘宝")
        self.comboBox.setFixedSize(70, 20)
        self.button1 = QPushButton('搜索', self)
        self.button1.clicked.connect(self.search)
        self.button1.setFixedSize(70, 30)
        self.button2 = QPushButton('保存数据', self)
        self.button2.clicked.connect(self.save_data)
        self.button2.setFixedSize(70, 30)
        self.button3 = QPushButton('自动评价', self)
        self.button3.clicked.connect(self.auto_evaluate)
        self.button3.setFixedSize(70, 30)
        self.button4 = QPushButton('返回主页', self)
        self.button4.clicked.connect(self.back)
        self.button4.setFixedSize(70, 30)

        buttonsLayout.addWidget(self.comboBox)
        buttonsLayout.addWidget(self.button1)
        buttonsLayout.addWidget(self.button2)
        buttonsLayout.addWidget(self.button3)
        buttonsLayout.addWidget(self.button4)
        buttonsLayout.setAlignment(Qt.AlignLeft)
        layout.addLayout(buttonsLayout)

        # 文本浏览器
        self.resultBox = QWebEngineView()
        layout.addWidget(self.resultBox)
        self.setLayout(layout)

    def search(self):

        url_dict = {
            '京东': 'https://search.jd.com/Search?keyword=',
            '淘宝': 'https://s.taobao.com/search?q='
        }
        goods = quote(self.inputLine.text())
        platform = self.comboBox.currentText()
        url = url_dict.get(platform, None) + goods
        if url is not None:
            self.thread = SpiderThread(url, self.platform_dict[platform])
            self.thread.signal.connect(self.search_complete_signal.emit)
            self.thread.start()

    def auto_evaluate(self):
        print("auto_evaluate")
        platform = self.comboBox.currentText()
        self.thread = EvalutionThread(self.platform_dict[platform])
        self.thread.signal.connect(self.auto_evaluate_complete_signal.emit)
        self.thread.start()

    def save_data(self):
        print("save_data")
        platform = self.comboBox.currentText()
        self.thread = SaveThread(self.platform_dict[platform])
        self.thread.signal.connect(self.save_data_complete_signal.emit)
        self.thread.start()

    def back(self):
        platform = self.comboBox.currentText()
        self.thread = BackThread(self.platform_dict[platform])
        self.thread.signal.connect(self.back_complete_signal.emit)
        self.thread.start()

    def show_message(self, text):
        msg = QMessageBox()
        msg.setText(text)
        msg.exec_()

    def update_result(self, status, df, flag=0):
        if df is not None:
            result_text = []
            star = lambda \
                    x: 1 if 0 <= x < 0.2 else 2 if 0.2 <= x < 0.4 else 3 if 0.4 <= x < 0.6 else 4 if 0.6 <= x < 0.8 else 5
            num_images = len(df)
            image_counter = 0
            clear_folder('temp/img')
            with ThreadPoolExecutor(max_workers=48) as executor:
                futures = {executor.submit(download_image,
                                           'http:' + row['图片'] if not row['图片'].startswith('http') else row['图片'],
                                           index): (index, row) for index, row in df.iterrows()}

                for future in as_completed(futures):
                    index, row = futures[future]
                    try:
                        success, image_file = future.result()
                        image_path_absolute = os.path.join(script_dir, image_file)
                        image_file = 'file:///{}'.format(image_path_absolute)
                        if success:
                            image_counter += 1
                            print("加载中 {}/{}".format(image_counter, num_images))
                            result_text.append('<div class="product-container">')
                            result_text.append('<div class="product-counter">{}/{}</div>'.format(image_counter, num_images))
                            result_text.append('<div class="product-info">')
                            result_text.append(
                                '<div class="product-image"><img src="{}" width="150" height="150"/></div>'.format(
                                    image_file))
                            result_text.append('<div class="product-details">')
                            result_text.append('<strong>商品：</strong>' + str(row['商品']) + '<br/>')
                            result_text.append('<strong>价格￥：</strong>' + str(row['价格']) + '<br/>')
                            result_text.append('<strong>店铺：</strong>' + str(row['店铺']) + '<br/>')
                            result_text.append('<strong>购买人数：</strong>' + str(row['购买人数']) + '<br/>')
                            result_text.append('<strong>发货地：</strong>' + str(row['发货地']) + '<br/>')
                            result_text.append('<strong>链接：</strong> <a href="' + (
                                str(row['详情页']) if 'http' in str(row['详情页']) else 'https://' + str(
                                    row['详情页'])) + '">点击前往</a><br/>')
                            if flag == 1: result_text.append(
                                '<strong>推荐指数:</strong>' + str(star(row['score_normalized'])) + '<br/>')
                            result_text.append('</div></div></div><hr>')

                            result_text.append('<style>')
                            result_text.append(
                                '.product-container { display: flex; flex-direction: column; align-items: flex; }')
                            result_text.append('.product-counter { text-align: left; margin-bottom: 10px; }')
                            result_text.append('.product-info { display: flex; }')
                            result_text.append('.product-image { margin-right: 10px; }')
                            result_text.append('.product-details { text-align: left; }')
                            result_text.append('</style>\n')
                    except Exception as e:
                        print(e)
            result_text = ''.join(result_text)
            print("加载完成")
            with open('temp/page.html', 'w', encoding='utf-8') as file:
                file.write(result_text)
            self.resultBox.load(QUrl.fromLocalFile(os.path.abspath('temp/page.html')))
        else:
            self.resultBox.setHtml('<p>{}</p>'.format(status))


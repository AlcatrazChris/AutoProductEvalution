import shutil
import requests
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def download_image(url, index):
    try:
        image_data = requests.get(url).content
        image_file = 'temp/img/{}.jpg'.format(index)
        with open(image_file, 'wb') as handler:
            handler.write(image_data)
            return True, image_file
    except Exception as e:
        print("下载图片失败：", e)
        return False, None


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('删除失败 %s. 问题: %s' % (file_path, e))



def generate_wordcloud(words_list, save_path):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if len(words_list) == 0:
        text = "无评论"
    else:
        filtered_words = [word for word in words_list if len(word) > 1]
        text = " ".join(filtered_words)
    wordcloud = WordCloud(font_path='font/msyhl.ttc', width=150, height=150).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

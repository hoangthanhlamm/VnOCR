import os


train_size = 13092
val_size = 3274
test_size = 4091

img_width = 170
img_height = 64
img_channel = 1

letters = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂĐÊÔƠƯÀẢÃẠÁẰẲẴẶẮẦẨẪẬẤÈẺẼẸÉỀỂỄỆẾÌỈĨỊÍÒỎÕỌÓỒỔỖỘỐỜỞỠỢỚÙỦŨỤÚỪỬỮỰỨỲỶỸỴÝ,."\''
word_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂĐÊÔƠƯÀẢÃẠÁẰẲẴẶẮẦẨẪẬẤÈẺẼẸÉỀỂỄỆẾÌỈĨỊÍÒỎÕỌÓỒỔỖỘỐỜỞỠỢỚÙỦŨỤÚỪỬỮỰỨỲỶỸỴÝ'

n_classes = len(letters) + 1
batch_size = 32
max_length = 18
pred_length = 40

n_epochs = 20

beam_width = 10

dir_path = os.path.dirname(__file__)
data_path = os.path.join(dir_path, 'data')
csv_path = os.path.join(data_path, 'csv')
checkpoint_path = os.path.join(data_path, 'checkpoints')

pretrained_model = os.path.join(data_path, 'models/vn_model.h5')

download_data_url = 'https://drive.google.com/uc?id=1dVO8yyqvyGVeWnQ78C5WYOdjCwaa7mUr'
download_model_url = 'https://drive.google.com/uc?id=1-jxcAlRsv5Dr67iZ414vVHFlfGDFq5aU'

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.utils import send_from_directory
import cv2 as cv
import tensorflow as tf
import numpy as np
import os

# try:
# 	import shutil
# 	shutil.rmtree('uploaded / image')
# 	cd uploaded
# 	mkdir image
# 	cd ..
# 	print()
# except:
# 	pass

model = tf.keras.models.load_model('model2.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/'

@app.route('/')
def upload_f():
	return render_template('index.php')

def finds(filename):
	images = []
	# vals = ['Rime', 'Rainbow', 'Glaze', 'Fogsmog', 'Dew', 'Frost', 'Rain', 'Hail', 'Lightning', 'Sandstorm', 'Snow'] 
	vals = ['Dew', 'Fogsmog', 'Frost', 'Glaze', 'Hail', 'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sandstorm', 'Snow']
	path = "uploaded/" + filename
	
	img = cv.imread(path)

	img = cv.resize(img, (256, 256), interpolation = cv.INTER_AREA)

	images.append(img)
	images = np.array(images)

	pred = model.predict(images)
	print(pred)
	return str(vals[np.argmax(pred)])

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds(f.filename)
		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		return render_template('result.php', ss = val)
if __name__ == '__main__':
	app.run()

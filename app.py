import os, glob, sys
from flask import Flask, send_from_directory, flash, render_template, request, send_file, redirect, url_for
from p2l import * 
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "/Users/Flamingo/Documents/Paper2LaTeX/examples"
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

app = Flask(__name__)
app.secret_key = "p2l"
app.debug = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('main.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if request.method == 'POST':
      if 'img' not in request.files:
          flash('No file selected')
          return redirect(url_for('main'))
      img = request.files['img']
      if img and allowed_file(img.filename):
          img_name = secure_filename(img.filename)
          img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
          img_nodes, img = get_semantics(img_name)
          bbox_edges = make_bbox_edge_dict(img_nodes)
          graph = find_edges(img, img_nodes, bbox_edges)
          transpile(graph)
          return redirect(url_for('uploaded_file', filename = img_name))
      
    else:
        return redirect(url_for('main'))

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

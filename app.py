import os
from flask import Flask, jsonify, request, json, render_template, redirect, send_from_directory
from lib.bruh import Bruh
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
JOB_POSTING_DIR = 'job_posting_dir'
RESUME_DIR = 'resume_dir'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER

# serve the resumes (viewable and downloadable in browser) based on the path name on flask server
@app.route('/uploads/resume_dir/<path:filename>')
def download(filename):
    return send_from_directory(directory="uploads/resume_dir", filename=filename)

# main function, which have the upload section and will later redirect to output section
@app.route('/', methods=['GET', 'POST'])
def analyse():
    if request.method == 'GET':
        return render_template('uploadForm.html')

    # create the necessary paths if not exist
    target = UPLOAD_FOLDER
    if not os.path.isdir('/'.join([target, JOB_POSTING_DIR])):
        os.mkdir('/'.join([target, JOB_POSTING_DIR]))
    if not os.path.isdir('/'.join([target, RESUME_DIR])):
        os.mkdir('/'.join([target, RESUME_DIR]))

    # save job posting
    if request.files['job_posting']:
        job_posting_file = request.files['job_posting']
        job_posting_fileName = request.files['job_posting'].filename
        job_posting_fileName = secure_filename(job_posting_fileName)
        job_posting_destination = "/".join([target, "job_posting_dir", job_posting_fileName])
        job_posting_file.save(job_posting_destination)

    # save resumes
    if request.files['resumes']:
        for resume in request.files.getlist('resumes'):
            resume_fileName = resume.filename
            if ".DS_Store" not in resume_fileName:
                resume_fileName = secure_filename(resume_fileName)
                resume_destination = "/".join([target, "resume_dir"]) + "/" + resume_fileName
                resume.save(resume_destination)

    bruh = Bruh('/'.join([target, RESUME_DIR]), '/'.join(['uploads', JOB_POSTING_DIR]))
    output = bruh.run_bert('roberta-large-nli-stsb-mean-tokens')
    print(output)

    # TODO
    # run Bert function here to start the job
    # output is in dictionary format as shown below, but with filename and respective scores
    # the corresponding file paths will be searched inside the resume_dir and the final result will be constructed
    '''
    output = {
        'Lecture_-_Classification.pdf': 10,
        '1b._big_data_lec1.pdf': 20
    }
    '''
    return render_template('output.html', output=output)

# delete all uploaded files after the job, will be integrated into main function later
@app.route('/destroy', methods=['DELETE'])
def destroy():
    job_posting_dir = '/'.join(['uploads', JOB_POSTING_DIR])
    resume_dir = '/'.join(['uploads', RESUME_DIR])

    if os.path.exists(job_posting_dir):
        for file in os.listdir(job_posting_dir):
            os.remove('/'.join(['uploads', JOB_POSTING_DIR, file]))
        os.rmdir(job_posting_dir)

    if os.path.exists(resume_dir):
        for file in os.listdir(resume_dir):
            os.remove('/'.join(['uploads', RESUME_DIR, file]))
        os.rmdir(resume_dir)

    return jsonify({"Message": "Successfully deletes all the uploaded files for current job"})

if __name__ == '__main__':
    app.run()

import re,string
import glob,os

text_files = glob.glob('./Dataset/train/pos/*.txt')


def clean_text(text):
	clean = re.sub(r'<\w*\s?/?>',' ',text)
	for punc in string.punctuation:
		clean = clean.replace(punc,' ')
	clean = re.sub(r'\s\s+',' ',clean)

	return clean

pos_exam = {}
for filename in text_files:
    with open(filename,"r") as fd:
        text = fd.read()
        pos_exam[clean_text(text)]=1

print(len(pos_exam))

text_files = glob.glob('./Dataset/train/neg/*.txt')

neg_exam = {}
for filename in text_files:
    with open(filename,"r") as fd:
        text = fd.read()
        neg_exam[clean_text(text)]=0

print(len(neg_exam))

pos_exam.update(neg_exam)
examples = pos_exam

	
# Writing it in a CSV file

with open('./Dataset/train/train.csv','w') as csvfile:
	for review, label in examples.items():
		csvfile.write("{},{}\n".format(review, label))
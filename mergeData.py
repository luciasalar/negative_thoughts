import pandas as pd 
import numpy as np
import xml.etree.ElementTree as ET
import re

dep = pd.read_csv('dep.csv')
status = pd.read_csv('user_status_match_dep.csv', names = ('num', 'userid', 'time','text','num2'))
demog = pd.read_csv('demog.csv')
like_dic = pd.read_csv('user_like_anonymous.csv')
big5 = pd.read_csv('big5.csv')
schwartz = pd.read_csv('svs.csv')
schwartz.columns = schwartz.columns.str.replace('id','userid')
swl = pd.read_csv('swl.csv')

status = status[['userid','time','text']]

#group text accroding to userid 1048 users 
status1 = status.groupby(['userid'])['text']
#status1.get_group()
#status2 = status1.apply(lambda x: str(x).lower())

status_unique = status.drop_duplicates(subset='userid')
like_unique = like_dic.drop_duplicates(subset = 'userid')

status_dep = pd.merge(status_unique, dep, on='userid', how = 'inner')
status_dep = status_dep.drop_duplicates(subset='userid')
status_dep_demog = pd.merge(status_dep, demog, on='userid')
status_dep_demog_like = pd.merge(status_dep_demog, like_unique, on='userid', how = 'inner') #1040
status_dep_demog_like_big5 = pd.merge(status_dep_demog_like, big5, on='userid', how ='inner') #975
status_dep_demog_like_big5_schwartz = pd.merge(status_dep_demog_like_big5, schwartz, on='userid', how ='inner')  
status_dep_demog_like_big5_schwartz = status_dep_demog_like_big5_schwartz.drop_duplicates(subset = 'userid') #558
status_dep_demog_like_big5_schwartz_swl = pd.merge(status_dep_demog_like_big5_schwartz, swl, on='userid', how ='inner') #395
status_dep_demog_like_big5_schwartz_swl = status_dep_demog_like_big5_schwartz_swl.drop_duplicates(subset = 'userid') #333


#status_dep_demog.to_csv("status_dep_demog_unique.csv")



#like_dic1 = like_dic[:10000000]
# like_dic2 = like_dic.groupby(['userid'])['like_id']

# status_dep_demoq_like = pd.merge(status_dep_demog,like_dic1, on = 'userid', how='inner')

# status_dep_demoq_like2 = status_dep_demoq_like[pd.notnull(status_dep_demoq_like['text'])]


#clean text
def sanitize(sent):
    words = str(sent).split()
    new_words = []
    # ps = PorterStemmer()
    
    for w in words:
        w = w.lower().replace("**bobsnewline**","")
        # remove non English word characters
        w = re.sub(r',',' quot ', w)
        # remove puncutation 
        w = re.sub(r'<',' lt ',w)
        # w = ps.stem(w)
        w = re.sub(r'>',' gt ',w)
        w = re.sub(r'&',' amp ',w)
        new_words.append(w)
        
    return ' '.join(new_words)

status_dep_demog_like_big5_schwartz_swl['text'] = status_dep_demog_like_big5_schwartz_swl['text'].apply(sanitize)
status['text'] = status['text'].apply(sanitize)

#convert variable to xml
# def func(row):
#     xml = ['<item>']
#     for field in row.index:
#         xml.append('  <field name="{0}">{1}</field>'.format(field, row[field]))
#     xml.append('</item>')
#     return '\n'.join(xml)

#convert variable to xml
def func_new(row):
	xml = []
	first = True
	for field in row.index:
		if first:
			xml.append('<item value="{0}">'.format(row[field]))
			first = False
		else:
			xml.append('  <field name="{0}">{1}</field>'.format(field, row[field]))
	xml.append('</item>')
	return '\n'.join(xml)

status_dep_demog_like_big5_schwartz_swl = ('\n'.join(status_dep_demog_like_big5_schwartz_swl.apply(func_new, axis=1)))

myfile = open("status_dep_demog_like_big5_schwartz_swl.xml", "w")  
myfile.write(status_dep_demog_like_big5_schwartz_swl)
myfile.close()


#parse with element tree
tree = ET.parse('status_dep_demog_like_big5_schwartz_swl.xml')

#print tree element
root = tree.getroot()
for i in root:
    for j in i:
    	if j.attrib['name'] == "text":
    		print(j.text)



for i in root:
	for j in i:
		if j.attrib['name'] == "text":
			


tree2 = ET.parse('status_dep_demog_like2.xml')

tmp = None
root2 = tree2.getroot()
for i in root2:
	if i.attrib['value'] == 'a36b5610cdc120eda6507c40adc597c8':
		for j in i:
			if j.attrib['name'] == "text":
				tmp = j
				print(j.text)

def find_text_node(tree, value):
	root2 = tree2.getroot()
	for i in root2:
		if i.attrib['value'] == value:
			for j in i:
				if j.attrib['name'] == "text":
					return j
	return None


for i in range(1, len(status['userid'])):
	userid = status['userid'][i]
	xmlNode = find_text_node(tree2, userid)
	if xmlNode is not None:
		xmlNode.text = xmlNode.text + " " + str(status['text'][i])

####append like_id

for i in root2:
	if i.attrib['value'] == 'a36b5610cdc120eda6507c40adc597c8':
		for j in i:
			if j.attrib['name'] == "like_id":
				tmp = j
				print(j.text)

def find_like_node(tree, value):
	root2 = tree2.getroot()
	for i in root2:
		if i.attrib['value'] == value:
			for j in i:
				if j.attrib['name'] == "like_id":
					return j
	return None



for i in range(1, len(status['userid'])):
	userid = status['userid'][i]
	xmlNode = find_like_node(tree2, userid)
	if xmlNode is not None:
		xmlNode.text = xmlNode.text + " " + str(like_dic['like_id'][i])

tree2.write('status_dep_demog_like.xml')

status = status.iloc[1:]

# reverse_map = {}                                                             
#      for i in range(len(status['userid'])):                       
#      	reverse_map[status['userid'][i]] = i

# for i in root2:
# 	userid = i.attrib['value']
# 	line_number = reverse_map[userid]
# 	for j in i:
# 		if j.attrib['name'] == "text":
# 			j.text = j.text + " " + status['text'][line_number]
# 			print(j.text)



# [i for i in status['userid']]
# status['text']
















